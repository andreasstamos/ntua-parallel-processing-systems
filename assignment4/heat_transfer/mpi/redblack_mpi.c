#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"

#define ISRED(_i,_j)   (((_i)+local[0]*rank_grid[0] + (_j)+local[1]*rank_grid[1])%2 == 0)
#define ISBLACK(_i,_j) (((_i)+local[0]*rank_grid[0] + (_j)+local[1]*rank_grid[1])%2 == 1)

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int i,j,t;

    #ifdef TEST_CONV
    int global_converged=0,converged=0; //flags for convergence, global and per process
    double tconv=0, conv_time;
    struct timeval tconvs, tconvf;
    #endif

    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory
    double omega; 			//relaxation factor - useless for Jacobi

    struct timeval tts,ttf,tcs,tcf;   //Timers: total-> tts,ttf, computation -> tcs,tcf
    double ttotal=0,tcomp=0,total_time,comp_time;
    
    double ** U, ** u_current, ** u_previous, ** swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous
    

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc!=5 && rank==0) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
	//----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator
		
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	                //rank mapping on the new communicator

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//
    
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
        }
    }

	//Initialization of omega
    omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    if (rank==0) {
        U=allocate2d(global_padded[0],global_padded[1]);   
        init2d(U,global[0],global[1]);
    }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous=allocate2d(local[0]+2,local[1]+2);
    u_current=allocate2d(local[0]+2,local[1]+2);   
       
    //----Distribute global 2D-domain from rank 0 to all processes----//
         
 	//----Appropriate datatypes are defined here----//
	/*****The usage of datatypes is optional*****/
    
    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
    MPI_Type_commit(&local_block);

    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int * scatteroffset, * scattercounts;
    if (rank==0) {
        scatteroffset=(int*)malloc(size*sizeof(int));
        scattercounts=(int*)malloc(size*sizeof(int));
        for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                scattercounts[i*grid[1]+j]=1;
                scatteroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
            }
    }


    //----Rank 0 scatters the global matrix----//

    MPI_Scatterv((rank==0 ? &(U[0][0]) : NULL), scattercounts, scatteroffset, global_block,
            &(u_current[1][1]), 1, local_block,
            0, CART_COMM);

    if (rank==0)
        free2d(U);

 
    //----Define datatypes or allocate buffers for message passing----//

    MPI_Datatype box_boundary_row_even, box_boundary_col_even;

    MPI_Type_vector((local[1]+1)/2, 1, 2, MPI_DOUBLE, &box_boundary_row_even);
    MPI_Type_commit(&box_boundary_row_even);

    MPI_Type_vector((local[0]+1)/2, 1, 2*(local[1]+2), MPI_DOUBLE, &box_boundary_col_even);
    MPI_Type_commit(&box_boundary_col_even);
     
    MPI_Datatype box_boundary_row_odd, box_boundary_col_odd;
    
    MPI_Type_vector(local[1]/2, 1, 2, MPI_DOUBLE, &box_boundary_row_odd);
    MPI_Type_commit(&box_boundary_row_odd);

    MPI_Type_vector(local[0]/2, 1, 2*(local[1]+2), MPI_DOUBLE, &box_boundary_col_odd);
    MPI_Type_commit(&box_boundary_col_odd);
    

    //----Find the 4 neighbors with which a process exchanges messages----//

    int north, south, east, west;

    MPI_Cart_shift(CART_COMM, 0, 1, &north, &south);
    MPI_Cart_shift(CART_COMM, 1, 1, &west,  &east);

    //---Define the iteration ranges per process-----//
    int i_min = 1, i_max = local[0], j_min = 1, j_max = local[1];

    if (north == MPI_PROC_NULL) {
        i_min = 2;
        for (int j=1;j<=local[1];++j) u_previous[1][j] = u_current[1][j];
    }
    if (south == MPI_PROC_NULL) {
        i_max = local[0]-1;
        for (int j=1;j<=local[1];++j) u_previous[local[0]][j] = u_current[local[0]][j];
    }
    if (west  == MPI_PROC_NULL) {
        j_min = 2;
        for (int i=1;i<=local[0];++i) u_previous[i][1]= u_current[i][1];
    }
    if (east  == MPI_PROC_NULL) {
        j_max = local[1]-1;
        for (int i=1;i<=local[0];++i) u_previous[i][local[1]] = u_current[i][local[1]];
    }
    
    /* Three types of ranges:
        -internal processes
        -boundary processes
        -boundary processes and padded global array
    */

    //----Computational core----//   
    gettimeofday(&tts, NULL);
    
    #ifdef TEST_CONV
    for (t=0;t<T && !global_converged;t++) {
    #endif
    #ifndef TEST_CONV
    #undef T
    #define T 256
    for (t=0;t<T;t++) {
    #endif
        MPI_Request r[8];

        swap = u_previous;
        u_previous = u_current;
        u_current = swap;

        /*Add appropriate timers for computation*/
        
        // Red-phase

        gettimeofday(&tcs, NULL);

        for (int i=i_min; i<=i_max; ++i) {
            for (int j=j_min; j<=j_max; ++j) {
                if ((i+j)%2==0)
                    u_current[i][j]=u_previous[i][j]+(omega/4.0)*(u_previous[i-1][j]+u_previous[i+1][j]+u_previous[i][j-1]+u_previous[i][j+1]-4*u_previous[i][j]);
            }
        }
        
        gettimeofday(&tcf, NULL);
        
        tcomp += (tcf.tv_sec - tcs.tv_sec) + (tcf.tv_usec - tcs.tv_usec) * 1.0e-6;

        // exchange red
        MPI_Isend(&u_current[1][1+ISBLACK(1,1)],                    1, ISRED(1, 1)          ? box_boundary_row_even : box_boundary_row_odd, north, 0, CART_COMM, &r[4]);
        MPI_Irecv(&u_current[local[0]+1][1+ISBLACK(local[0]+1, 1)], 1, ISRED(local[0]+1, 1) ? box_boundary_row_even : box_boundary_row_odd, south, 0, CART_COMM, &r[0]);

        MPI_Isend(&u_current[local[0]][1+ISBLACK(local[0], 1)],     1, ISRED(local[0], 1)   ? box_boundary_row_even : box_boundary_row_odd, south, 1, CART_COMM, &r[5]);
        MPI_Irecv(&u_current[0][1+ISBLACK(0, 1)],                   1, ISRED(0, 1)          ? box_boundary_row_even : box_boundary_row_odd, north, 1, CART_COMM, &r[1]);

        MPI_Isend(&u_current[1+ISBLACK(1,1)][1],                    1, ISRED(1, 1)          ? box_boundary_col_even : box_boundary_col_odd,  west, 2, CART_COMM, &r[6]);
        MPI_Irecv(&u_current[1+ISBLACK(1, local[1]+1)][local[1]+1], 1, ISRED(1, local[1]+1) ? box_boundary_col_even : box_boundary_col_odd,  east, 2, CART_COMM, &r[2]);

        MPI_Isend(&u_current[1+ISBLACK(1, local[1])][local[1]],     1, ISRED(1, local[1])   ? box_boundary_col_even : box_boundary_col_odd,  east, 3, CART_COMM, &r[7]);
        MPI_Irecv(&u_current[1+ISBLACK(1, 0)][0],                   1, ISRED(1, 0)          ? box_boundary_col_even : box_boundary_col_odd,  west, 3, CART_COMM, &r[3]);


        MPI_Waitall(8, r, MPI_STATUSES_IGNORE);

        // Black-phase

        gettimeofday(&tcs, NULL);

        for (int i=i_min; i<=i_max; ++i) {
            for (int j=j_min; j<=j_max; ++j) {
                if ((i+j)%2==1)
                    u_current[i][j]=u_previous[i][j]+(omega/4.0)*(u_current[i-1][j]+u_current[i+1][j]+u_current[i][j-1]+u_current[i][j+1]-4*u_previous[i][j]);
            }
        }
        
        gettimeofday(&tcf, NULL);
        
        tcomp += (tcf.tv_sec - tcs.tv_sec) + (tcf.tv_usec - tcs.tv_usec) * 1.0e-6;

        // exchange black
        MPI_Isend(&u_current[1][1+ISRED(1,1)],                    1, ISBLACK(1, 1)          ? box_boundary_row_even : box_boundary_row_odd, north, 0, CART_COMM, &r[4]);
        MPI_Irecv(&u_current[local[0]+1][1+ISRED(local[0]+1, 1)], 1, ISBLACK(local[0]+1, 1) ? box_boundary_row_even : box_boundary_row_odd, south, 0, CART_COMM, &r[0]);

        MPI_Isend(&u_current[local[0]][1+ISRED(local[0], 1)],     1, ISBLACK(local[0], 1)   ? box_boundary_row_even : box_boundary_row_odd, south, 1, CART_COMM, &r[5]);
        MPI_Irecv(&u_current[0][1+ISRED(0, 1)],                   1, ISBLACK(0, 1)          ? box_boundary_row_even : box_boundary_row_odd, north, 1, CART_COMM, &r[1]);

        MPI_Isend(&u_current[1+ISRED(1,1)][1],                    1, ISBLACK(1, 1)          ? box_boundary_col_even : box_boundary_col_odd,  west, 2, CART_COMM, &r[6]);
        MPI_Irecv(&u_current[1+ISRED(1, local[1]+1)][local[1]+1], 1, ISBLACK(1, local[1]+1) ? box_boundary_col_even : box_boundary_col_odd,  east, 2, CART_COMM, &r[2]);

        MPI_Isend(&u_current[1+ISRED(1, local[1])][local[1]],     1, ISBLACK(1, local[1])   ? box_boundary_col_even : box_boundary_col_odd,  east, 3, CART_COMM, &r[7]);
        MPI_Irecv(&u_current[1+ISRED(1, 0)][0],                   1, ISBLACK(1, 0)          ? box_boundary_col_even : box_boundary_col_odd,  west, 3, CART_COMM, &r[3]);

        MPI_Waitall(8, r, MPI_STATUSES_IGNORE);

        #ifdef TEST_CONV
        if (t%C==0) {
            /*Test convergence*/

            gettimeofday(&tconvs, NULL);

            converged = converge(u_previous, u_current, 1, local[0], 1, local[1]);
            MPI_Allreduce(&converged, &global_converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

            gettimeofday(&tconvf, NULL);
            
            tconv += (tconvf.tv_sec - tconvs.tv_sec) + (tconvf.tv_usec - tconvs.tv_usec) * 1.0e-6;
        }		
        #endif

    }

    gettimeofday(&ttf,NULL);

    ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;

    MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    #ifdef TEST_CONV
    MPI_Reduce(&tconv,&conv_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    #endif


    //----Rank 0 gathers local matrices back to the global matrix----//
   
    if (rank==0) {
            U=allocate2d(global_padded[0],global_padded[1]);
    }

    MPI_Gatherv(&u_current[1][1], 1, local_block, 
            (rank == 0 ? &U[0][0] : NULL), scattercounts, scatteroffset, global_block,
            0, CART_COMM);


    //----Printing results----//

    if (rank==0) {
        printf("RedBlackSOR X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n",global[0],global[1],grid[0],grid[1],t,comp_time,total_time,U[global[0]/2][global[1]/2]);
	
        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"resRedBlackSORMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
        free(s);
        #endif

        #ifdef TEST_CONV
        printf("ConvTime %lf\n", conv_time);
        #endif

    }

    if (rank == 0) {
        free(scatteroffset);
        free(scattercounts);
    }

    free2d(u_previous);
    free2d(u_current);

    MPI_Type_free(&global_block);
    MPI_Type_free(&local_block);
    MPI_Type_free(&dummy);
    MPI_Type_free(&box_boundary_row_even);
    MPI_Type_free(&box_boundary_col_even);
    MPI_Type_free(&box_boundary_row_odd);
    MPI_Type_free(&box_boundary_col_odd);
    
    if (rank == 0) free2d(U);

    MPI_Finalize();


    return 0;
}
