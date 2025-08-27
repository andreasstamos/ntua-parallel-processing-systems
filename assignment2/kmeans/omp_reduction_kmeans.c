#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <omp.h>
#include <numa.h>
#include "kmeans.h"

#define TIMING_FD 999


// square of Euclid distance between two multi-dimensional points
inline static double euclid_dist_2(int    numdims,  /* no. dimensions */
								 double * coord1,   /* [numdims] */
								 double * coord2)   /* [numdims] */
{
	int i;
	double ans = 0.0;

	for(i=0; i<numdims; i++)
		ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

	return ans;
}

inline static int find_nearest_cluster(int      numClusters, /* no. clusters */
									   int      numCoords,   /* no. coordinates */
									   double * object,      /* [numCoords] */
									   double * clusters)    /* [numClusters][numCoords] */
{
	int index, i;
	double dist, min_dist;

	// find the cluster id that has min distance to object 
	index = 0;
	min_dist = euclid_dist_2(numCoords, object, clusters);

	for(int i=1; i<numClusters; i++) {
		dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
		// no need square root 
		if (dist < min_dist) { // find the min and its array index
			min_dist = dist;
			index    = i;
		}
	}
	return index;
}

void kmeans(double * objects,          /* in: [numObjs][numCoords] */
			int      numCoords,        /* no. coordinates */
			int      numObjs,          /* no. objects */
			int      numClusters,      /* no. clusters */
			double   threshold,        /* minimum fraction of objects that change membership */
			long     loop_threshold,   /* maximum number of iterations */
			int    * membership,       /* out: [numObjs] */
			double * clusters)         /* out: [numClusters][numCoords] */
{
	int loop=0;
	double timing = 0;

	double delta;		// fraction of objects whose clusters change in each loop 
	int * newClusterSize;	// [numClusters]: no. objects assigned in each new cluster 
	double * newClusters;	// [numClusters][numCoords] 
	int nthreads;		// no. threads 

	nthreads = omp_get_max_threads();
	printf("OpenMP Kmeans - Reduction\t(number of threads: %d)\n", nthreads);

	// initialize membership
	for (int i=0; i<numObjs; i++)
		membership[i] = -1;

	// initialize newClusterSize and newClusters to all 0 
	newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
	newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

	// Each thread calculates new centers using a private space. After that, thread 0 does an array reduction on them.
	int * local_newClusterSize[nthreads];	// [nthreads][numClusters] 
	double * local_newClusters[nthreads];	// [nthreads][numClusters][numCoords]

	/*
	 * Hint for false-sharing
	 * This is noticed when numCoords is low (and neighboring local_newClusters exist close to each other).
	 * Allocate local cluster data with a "first-touch" policy.
	 */
	// Initialize local (per-thread) arrays (and later collect result on global arrays)
        #pragma omp parallel for schedule(static)
	for (int k=0; k<nthreads; k++)
	{
		int tid = omp_get_thread_num();
		local_newClusterSize[tid] = (typeof(*local_newClusterSize)) numa_alloc_local(numClusters * sizeof(**local_newClusterSize));
		local_newClusters[tid] = (typeof(*local_newClusters)) numa_alloc_local(numClusters * numCoords * sizeof(**local_newClusters));
	}

	timing = wtime();
	do {
		// before each loop, set cluster data to 0
		for (int i=0; i<numClusters; i++) {
			for (int j=0; j<numCoords; j++)
				newClusters[i*numCoords + j] = 0.0;
			newClusterSize[i] = 0;
		}

		delta = 0.0;

                #pragma omp parallel for schedule(static) reduction(+:delta)
		for (int i=0; i<numObjs; i++)
		{
			// find the array index of nearest cluster center 
			int index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
			
			// if membership changes, increase delta by 1 
			if (membership[i] != index)
				delta += 1.0;
			
			// assign the membership to object i 
			membership[i] = index;
			
			// update new cluster centers : sum of all objects located within (average will be performed later)
			int tid = omp_get_thread_num();
			local_newClusterSize[tid][index]++;
			for (int j=0; j<numCoords; j++)
				local_newClusters[tid][index*numCoords + j] += objects[i*numCoords + j];

		}

		for (int k=0; k<nthreads; ++k) {
			for (int i=0; i<numClusters; ++i) {
				for (int j=0; j<numCoords; ++j) {
					newClusters[i*numCoords + j] += local_newClusters[k][i*numCoords + j];
				}
				newClusterSize[i] += local_newClusterSize[k][i];
			}
		}

		// average the sum and replace old cluster centers with newClusters 
		for (int i=0; i<numClusters; i++) {
			if (newClusterSize[i] > 0) {
				for (int j=0; j<numCoords; j++) {
					clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
				}
			}
		}

		// Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
		delta /= numObjs;

		loop++;
		printf("\r\tcompleted loop %d", loop);
		fflush(stdout);
	} while (delta > threshold && loop < loop_threshold);
	timing = wtime() - timing;

	// write times to a file opened by the runner script. this is to make plotting easier.
	if (fcntl(TIMING_FD, F_GETFD) != -1) dprintf(TIMING_FD,"%d %lf %d\n", nthreads, timing, loop);
	
	printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);

	for (int k=0; k<nthreads; k++)
	{
		numa_free(local_newClusterSize[k], numClusters * sizeof(**local_newClusterSize));
		numa_free(local_newClusters[k], numClusters * numCoords * sizeof(**local_newClusters));
	}
	free(newClusters);
	free(newClusterSize);
}
