/****************************************************** 
 ************* Conway's game of life ******************
 ******************************************************

 Usage: ./exec ArraySize TimeSteps                   

 Compile with -DOUTPUT to print output in output.gif 
 (You will need ImageMagick for that - Install with
 sudo apt-get install imagemagick)
 WARNING: Do not print output for large array sizes!
 or multiple time steps!
 ******************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include<vector>
using namespace std;

#define FINALIZE "\
convert -delay 20 `ls -1 out*.pgm | sort -V` output.gif\n\
rm *pgm\n\
"

void init_random(vector< vector<bool> > &array1, vector< vector<bool> > &array2, int N);
#ifdef OUTPUT
void print_to_pgm( unsigned char ** array, int N, int t );
#endif

int main (int argc, char * argv[]) {
	int N;	 			//array dimensions
	int T; 				//time steps

	double time;			//variables for timing
	struct timeval ts,tf;

	/*Read input arguments*/
	if ( argc != 3 ) {
		fprintf(stderr, "Usage: ./exec ArraySize TimeSteps\n");
		exit(-1);
	}
	else {
		N = atoi(argv[1]);
		T = atoi(argv[2]);
	}

	/*Allocate and initialize matrices*/
	vector < vector<bool> > current(N, vector<bool>(N));
	vector < vector<bool> > previous(N, vector<bool>(N));

	init_random(previous, current, N);		//initialize previous array with pattern

	#ifdef OUTPUT
	print_to_pgm(previous, N, 0);
	#endif

	/*Game of Life*/

	gettimeofday(&ts,NULL);
	for (int t = 0 ; t < T ; t++ ) {
		#pragma omp parallel for schedule(static)
		for (int i = 1 ; i < N-1 ; i++ )
			for (int j = 1 ; j < N-1 ; j++ ) {
				int nbrs = previous[i+1][j+1] + previous[i+1][j] + previous[i+1][j-1] \
				       + previous[i][j-1] + previous[i][j+1] \
				       + previous[i-1][j-1] + previous[i-1][j] + previous[i-1][j+1];
				current[i][j] = nbrs == 3 || (previous[i][j]+nbrs == 3);
			}

		#ifdef OUTPUT
		print_to_pgm(current, N, t+1);
		#endif
		//Swap current array with previous array 
		current.swap(previous);
	}
	gettimeofday(&tf,NULL);
	time=(tf.tv_sec-ts.tv_sec)+(tf.tv_usec-ts.tv_usec)*0.000001;

	printf("GameOfLife: Size %d Steps %d Time %lf\n", N, T, time);
	#ifdef OUTPUT
	system(FINALIZE);
	#endif
}

void init_random(vector< vector<bool> > &array1, vector< vector<bool> > &array2, int N) {
	int i,pos,x,y;

	for ( i = 0 ; i < (N * N)/10 ; i++ ) {
		pos = rand() % ((N-2)*(N-2));
		array1[pos%(N-2)+1][pos/(N-2)+1] = 1;
		array2[pos%(N-2)+1][pos/(N-2)+1] = 1;

	}
}

#ifdef OUTPUT
void print_to_pgm(unsigned char ** array, int N, int t) {
	int i,j;
	char * s = malloc(30*sizeof(char));
	sprintf(s,"out%d.pgm",t);
	FILE * f = fopen(s,"wb");
	fprintf(f, "P5\n%d %d 1\n", N,N);
	for ( i = 0; i < N ; i++ ) 
		for ( j = 0; j < N ; j++)
			if ( array[i][j]==1 )
				fputc(1,f);
			else
				fputc(0,f);
	fclose(f);
	free(s);
}
#endif

