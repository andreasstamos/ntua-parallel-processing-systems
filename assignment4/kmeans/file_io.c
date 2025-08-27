#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include <mpi.h>

#include "kmeans.h"

#define MAX(x,y) ((x)<(y) ? (y) : (x))

double * dataset_generation(int numObjs, int numCoords, long *rank_numObjs)
{
    double * objects = NULL, * rank_objects = NULL;
    long i, j, k;

    // Random values that will be generated will be between 0 and 10.
    double val_range = 10;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate number of objects that each rank will examine (*rank_numObjs)
    // a number of (numObjs % size) ranks will have to examine just one more object. (for simplicity these will be the lowered numbered ranks)
    *rank_numObjs = numObjs/size;

    /* allocate space for objects[][] and read all objects */
    int sendcounts[size], displs[size];
    if (rank == 0) {
        objects = (typeof(objects)) malloc(numObjs * numCoords * sizeof(*objects));
        
        // Calculate sendcounts and displs, which will be used to scatter data to each rank.

        displs[0] = 0;
        sendcounts[0] = numCoords * (numObjs % size >= 1 ? *rank_numObjs+1 : *rank_numObjs);
        for (int i=1; i<numObjs % size; ++i) {
            displs[i] = displs[i-1] + sendcounts[i-1];
            sendcounts[i] = numCoords * (*rank_numObjs + 1);
        }
        for (int i=MAX(1, numObjs%size); i<size; ++i) {
            displs[i] = displs[i-1] + sendcounts[i-1];
            sendcounts[i] = numCoords * (*rank_numObjs);
        }
    }


    // Broadcasting of sendcounts and displs is not necessary. Each rank, based on its rank number, can calculate if it has been assigned
    // *rank_numObjs or *rank_numObjs+1

    if (rank<numObjs%size) ++(*rank_numObjs);

    /* allocate space for objects[][] (for each rank separately) and read all objects */
    rank_objects = (typeof(rank_objects)) malloc((*rank_numObjs) * numCoords * sizeof(*rank_objects));
    
    /* rank 0 will generate data for the objects array. This array will be used later to scatter data to each rank. */
    if (rank == 0) {
        for (i=0; i<numObjs; i++)
        {
            unsigned int seed = i;
            for (j=0; j<numCoords; j++)
            {
                objects[i*numCoords + j] = (rand_r(&seed) / ((double) RAND_MAX)) * val_range;
                if (_debug && i == 0)
                    printf("object[i=%ld][j=%ld]=%f\n",i,j,objects[i*numCoords + j]);
            }
        }
    }

    // Scatter objects to every rank. (hint: each rank may receive different number of objects)
    MPI_Scatterv(objects, sendcounts, displs, MPI_DOUBLE, rank_objects, numCoords * (*rank_numObjs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
        free(objects);

    return rank_objects;
}
