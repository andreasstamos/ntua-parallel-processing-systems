#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid() {
  return blockIdx.x * blockDim.x + threadIdx.x; /* Calculate 1-Dim global ID of a thread */
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
double euclid_dist_2_transpose(int numCoords,
                               int numObjs,
                               int numClusters,
                               double *objects,     // [numCoords][numObjs]
                               double *clusters,    // [numCoords][numClusters]
                               int objectId,
                               int clusterId) {
  double ans = 0.0;

  /* Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters, but for column-base format!!! */
  for (int i=0; i<numCoords; ++i) {
	  double diff = objects[i*numObjs+objectId]-clusters[i*numClusters+clusterId];
	  ans += diff*diff;
  }
 
  return (ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *deviceobjects,           //  [numCoords][numObjs]
                          double *deviceClusters,    //  [numCoords][numClusters]
                          double *devicenewClusters,
                          int *devicenewClusterSize,           //  [numClusters]
                          int *deviceMembership,
                          double *devdelta) {
  extern __shared__ char shmemBlock[];
  double *shmemClusters = reinterpret_cast<double*>(shmemBlock);
  double *shmemDelta = &shmemClusters[numClusters*numCoords];
  int *shmemClusterSizes = reinterpret_cast<int*>(&shmemDelta[1]);


  /* Copy deviceClusters to shmemClusters so they can be accessed faster.
    BEWARE: Make sure operations is complete before any thread continues... */
  for (int i=threadIdx.x; i< numClusters*numCoords;i+= blockDim.x) {
    shmemClusters[i] = deviceClusters[i];
  }
  if (threadIdx.x == 0) *shmemDelta = 0;

  __syncthreads();

  /* Get the global ID of the thread. */
  int tid = get_tid();
  int index;

  if (tid < numObjs) {
    double dist, min_dist;

    /* find the cluster id that has min distance to object */
    index = 0;
    min_dist = INFINITY;

    for (int i = 0; i < numClusters; i++) {
      dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, deviceobjects, shmemClusters, tid, i);

      /* no need square root */
      if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index = i;
      }
    }

    if (deviceMembership[tid] != index) {
      atomicAdd(shmemDelta, 1.0);
    }
    deviceMembership[tid] = index;

  }

  // update centroids phase

  for (int i=threadIdx.x; i<numClusters; i += blockDim.x) shmemClusterSizes[i] = 0;
  
  __syncthreads(); //before clearing shmemClusters, data must have been read by all threads.
    
  for (int i=threadIdx.x; i<numClusters*numCoords; i+= blockDim.x) shmemClusters[i] = 0;
  
  __syncthreads();

  if (tid < numObjs) {
    atomicAdd(&shmemClusterSizes[index], 1);
    for (int i=0; i<numCoords; ++i) atomicAdd(&shmemClusters[i*numClusters+index], deviceobjects[i*numObjs+tid]);
  }

  __syncthreads();

  for (int i=threadIdx.x; i<numCoords*numClusters; i+=blockDim.x) atomicAdd(&devicenewClusters[i], shmemClusters[i]);
  for (int i=threadIdx.x; i<numClusters; i+=blockDim.x) atomicAdd(&devicenewClusterSize[i], shmemClusterSizes[i]);
  
  if (threadIdx.x==0) atomicAdd(devdelta, *shmemDelta);

}

__global__ static
void update_centroids(int numCoords,
                      int numClusters,
                      int *devicenewClusterSize,           //  [numClusters]
                      double *devicenewClusters,           //  [numCoords][numClusters]
                      double *deviceClusters)              //  [numCoords][numClusters])
{
  /* average the sum and replace old cluster centers with newClusters */
  extern __shared__ int shmemClusterSizes[];
  
  int idx = get_tid();

  for (int i=threadIdx.x; i< numClusters; i+=blockDim.x)
    shmemClusterSizes[i] = devicenewClusterSize[i];

  __syncthreads();
  
  if (idx < numClusters) devicenewClusterSize[idx] = 0;

  if (idx < numCoords*numClusters) {
    int clusterSize = shmemClusterSizes[idx%numClusters];
    if (clusterSize>0) deviceClusters[idx] = devicenewClusters[idx] / clusterSize;
    devicenewClusters[idx] = 0;
  }

}

#define CPU_BLOCK_SIZE_SQRT 32


//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
void kmeans_gpu(double *objects,      /* in: [numObjs][numCoords] */
                int numCoords,    /* no. features */
                int numObjs,      /* no. objects */
                int numClusters,  /* no. clusters */
                double threshold,    /* % objects change membership */
                long loop_threshold,   /* maximum number of iterations */
                int *membership,   /* out: [numObjs] */
                double *clusters,   /* out: [numClusters][numCoords] */
                int blockSize) {
  double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0;
  double timing_gpu, timing_cpu, timing_transfers, transfers_time = 0.0, cpu_time = 0.0, gpu_time = 0.0;
  int loop_iterations = 0;
  int i, j, index, loop = 0;
  double delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
  /* Transpose dims */
  double **dimObjects = (double**) calloc_2d(numCoords, numObjs, sizeof(double));
  double **dimClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));
  double **newClusters = (double**) calloc_2d(numCoords, numClusters, sizeof(double));



  printf("\n|-----------Full-offload GPU Kmeans------------|\n\n");

  // Copy objects given in [numObjs][numCoords] layout to new
  //  [numCoords][numObjs] layout
  for (int i=0; i<numObjs; i+=CPU_BLOCK_SIZE_SQRT) {
    for (int j=0; j<numCoords; j+=CPU_BLOCK_SIZE_SQRT) {
      for (int bi=i; bi-i<CPU_BLOCK_SIZE_SQRT && bi<numObjs; ++bi) {
        for (int bj=j; bj-j<CPU_BLOCK_SIZE_SQRT && bj<numCoords; ++bj) {
          dimObjects[bj][bi] = objects[bi*numCoords+bj];
        }
      }
    }
  }


  double *deviceObjects;
  double *deviceClusters, *devicenewClusters;
  int *deviceMembership;
  int *devicenewClusterSize; /* [numClusters]: no. objects assigned in each new cluster */

  /* pick first numClusters elements of objects[] as initial cluster centers*/
  for (i = 0; i < numCoords; i++) {
    for (j = 0; j < numClusters; j++) {
      dimClusters[i][j] = dimObjects[i][j];
    }
  }

  /* initialize membership[] */
  for (i = 0; i < numObjs; i++) membership[i] = -1;

  timing = wtime() - timing;
  printf("t_alloc: %lf ms\n\n", 1000 * timing);
  timing = wtime();
  const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize) ? blockSize : numObjs;
  const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock; /* Calculate Grid size, e.g. number of blocks. */
  /*	Define the shared memory needed per block.
      - BEWARE: We can overrun our shared memory here if there are too many
      clusters or too many coordinates!
      - This can lead to occupancy problems or even inability to run.
      - Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
  const unsigned int clusterBlockSharedDataSize = numCoords*numClusters*sizeof(double) + numClusters*sizeof(int) + 1;


  cudaDeviceProp deviceProp;
  int deviceNum;
  cudaGetDevice(&deviceNum);
  cudaGetDeviceProperties(&deviceProp, deviceNum);

  if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
    error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
  }

  checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(double)));
  checkCuda(cudaMalloc(&deviceClusters, numClusters * numCoords * sizeof(double)));
  checkCuda(cudaMalloc(&devicenewClusters, numClusters * numCoords * sizeof(double)));
  checkCuda(cudaMalloc(&devicenewClusterSize, numClusters * sizeof(int)));
  checkCuda(cudaMalloc(&deviceMembership, numObjs * sizeof(int)));
  checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));

  timing = wtime() - timing;
  printf("t_alloc_gpu: %lf ms\n\n", 1000 * timing);
  timing = wtime();

  checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
                       numObjs * numCoords * sizeof(double), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(deviceMembership, membership,
                       numObjs * sizeof(int), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                       numClusters * numCoords * sizeof(double), cudaMemcpyHostToDevice));
  checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters * sizeof(int)));
  checkCuda(cudaMemset(devicenewClusters, 0, numClusters * numCoords * sizeof(double)));
  free(dimObjects[0]);

  timing = wtime() - timing;
  printf("t_get_gpu: %lf ms\n\n", 1000 * timing);
  timing = wtime();

  do {
    timing_internal = wtime();
    checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));
    timing_gpu = wtime();
    //printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);

    find_nearest_cluster
        <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
        (numCoords, numObjs, numClusters,
         deviceObjects, deviceClusters, devicenewClusters, devicenewClusterSize, deviceMembership, dev_delta_ptr);
  
    cudaDeviceSynchronize();
    checkLastCudaError();

    gpu_time += wtime() - timing_gpu;

    //printf("Kernels complete for itter %d, updating data in CPU\n", loop);

    timing_transfers = wtime();
    /* Copy dev_delta_ptr to &delta */
    checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(double), cudaMemcpyDeviceToHost));

    transfers_time += wtime() - timing_transfers;

    int update_centroids_block_sz;
    int update_centroids_dim_sz;
    const int update_centroids_shmem_sz = numClusters*sizeof(int);
   
    cudaOccupancyMaxPotentialBlockSize(&update_centroids_dim_sz, &update_centroids_block_sz, update_centroids, update_centroids_shmem_sz, numClusters*numCoords);

    update_centroids_dim_sz = (numClusters*numCoords + update_centroids_block_sz - 1) / update_centroids_block_sz;
    
    timing_gpu = wtime();
    update_centroids<<< update_centroids_dim_sz, update_centroids_block_sz, update_centroids_shmem_sz >>>
            (numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);
    cudaDeviceSynchronize();
    checkLastCudaError();
    gpu_time += wtime() - timing_gpu;

    timing_cpu = wtime();
    delta /= numObjs;
    //printf("delta is %f - ", delta);
    loop++;
    //printf("completed loop %d\n", loop);
    cpu_time += wtime() - timing_cpu;

    timing_internal = wtime() - timing_internal;
    if (timing_internal < timer_min) timer_min = timing_internal;
    if (timing_internal > timer_max) timer_max = timing_internal;
  } while (delta > threshold && loop < loop_threshold);

  checkCuda(cudaMemcpy(membership, deviceMembership,
                       numObjs * sizeof(int), cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                       numClusters * numCoords * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i=0; i<numCoords; i+=CPU_BLOCK_SIZE_SQRT) {
    for (int j=0; j<numClusters; j+=CPU_BLOCK_SIZE_SQRT) {
      for (int bi=i; bi-i<CPU_BLOCK_SIZE_SQRT && bi<numCoords; ++bi) {
        for (int bj=j; bj-j<CPU_BLOCK_SIZE_SQRT && bj<numClusters; ++bj) {
          clusters[bj*numCoords+bi] = dimClusters[bi][bj];
        }
      }
    }
  }

  timing = wtime() - timing;
  printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\t"
         "-> t_cpu_avg = %lf ms\n\t-> t_gpu_avg = %lf ms\n\t-> t_transfers_avg = %lf ms\n\n|-------------------------------------------|\n",
         loop, 1000 * timing, 1000 * timing / loop, 1000 * timer_min, 1000 * timer_max,
         1000 * cpu_time / loop, 1000 * gpu_time / loop, 1000 * transfers_time / loop);

  char outfile_name[1024] = {0};
  sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv",
          numObjs * numCoords * sizeof(double) / (1024 * 1024), numCoords, numClusters);
  FILE *fp = fopen(outfile_name, "a+");
  if (!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name);
  fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU", blockSize, timing / loop, timer_min, timer_max);
  fclose(fp);

  checkCuda(cudaFree(deviceObjects));
  checkCuda(cudaFree(deviceClusters));
  checkCuda(cudaFree(devicenewClusters));
  checkCuda(cudaFree(devicenewClusterSize));
  checkCuda(cudaFree(deviceMembership));

  return;
}

