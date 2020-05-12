/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 29, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm
 */

#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include "gauss_eliminate.h"


#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_omp(Matrix, int);
void gauss_eliminate_using_pthreads(Matrix, int);
void *compute_silver(void *args);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

/* Structure to pass arguments into various threads */
typedef struct args_for_thread
{
	int	tid;
	int	num_threads;
	int	num_elements;
        float	*U;
	int	rows;
	int	start;
	int	stop;

} ARGS_FOR_THREAD;


pthread_barrier_t barrierp;

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt_p;			                                        /* Upper triangular matrix computed by pthreads */
    Matrix U_mt_o;

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt_p = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */
    U_mt_o = allocate_matrix (matrix_size, matrix_size, 0);

    /* Copy contents A matrix into U matrices */
    int i, j, k, size, res;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt_p.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
	    U_mt_o.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.3f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    fprintf(stderr, "---------------------------------------------------\n");
    /* FIXME: Perform Gaussian elimination using OpenMP. 
     * The resulting upper triangular matrix should be returned in U_mt */
    for (i = 4; i <= 32; i = i*2){
 	fprintf(stderr, "\nPerforming gaussian elimination using %i pthreads\n", i);
	gettimeofday(&start, NULL);
	gauss_eliminate_using_pthreads(U_mt_p, i);
    	gettimeofday(&stop, NULL);
	fprintf(stderr, "CPU run time with %d pthreads = %0.3f s\n", i, (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

	/* Check if pthread result matches reference solution within specified tolerance */

	fprintf(stderr, "Checking results\n");
	size = matrix_size * matrix_size;
   	res = check_results(U_reference.elements, U_mt_p.elements, size, 1e-6);
    	fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");
	

    	fprintf(stderr, "\nPerforming gaussian elimination using omp and %d threads\n", i);
    	gettimeofday(&start, NULL);
    	gauss_eliminate_using_omp(U_mt_o, i);
    	gettimeofday(&stop, NULL);
    	fprintf(stderr, "CPU run time with %d threads = %0.3f s\n", i, (float)(stop.tv_sec - start.tv_sec\
    	            + (stop.tv_usec - start.tv_usec) / (float)1000000));

    	/* Check if pthread result matches reference solution within specified tolerance */
    	fprintf(stderr, "Checking results\n");
    	size = matrix_size * matrix_size;
    	res = check_results(U_reference.elements, U_mt_o.elements, size, 1e-6);
    	fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");
	fprintf(stderr, "---------------------------------------------------\n");

	/* Reset matrices to test with next thread amount */
	for (k = 0; k < A.num_rows; k++) {
        	for (j = 0; j < A.num_rows; j++) {
            		U_mt_p.elements[A.num_rows * k + j] = A.elements[A.num_rows * k + j];
	    		U_mt_o.elements[A.num_rows * k + j] = A.elements[A.num_rows * k + j];
        	}
    	}
    }

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt_p.elements);
    free(U_mt_o.elements);


    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using omp */
void gauss_eliminate_using_omp(Matrix U, int num_threads)
{
	int i, j, k, l;
	int num_elements = U.num_rows;
	float *U_elements = U.elements; 
	omp_set_num_threads(num_threads);
		
	for (k = 0; k < num_elements; k++) {
		
		#pragma omp parallel for default(none) shared(num_elements, k, U_elements) private(j) 
        	for (j = (k + 1); j < num_elements; j++) {   /* Reduce the current row. */
        	    if (U_elements[num_elements * k + k] == 0) {
        	        printf("Numerical instability. The principal diagonal element is zero.\n");
        	    }            
        	    U_elements[num_elements * k + j] = (float)(U_elements[num_elements * k + j] / U_elements[num_elements * k + k]);	/* Division step */
        	}
			
		U_elements[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 

		#pragma omp parallel for default(none) shared(num_elements, k, l, U_elements) private(i) 
	        for (i = (k + 1); i < num_elements; i++) {
        	    for (l = (k + 1); l < num_elements; l++)
	                U_elements[num_elements * i + l] = U_elements[num_elements * i + l] - (U_elements[num_elements * i + k] * U_elements[num_elements * k + l]);	/* Elimination step */
	            
	            U_elements[num_elements * i + k] = 0;
        	}
    	}	
	return;
}

/* Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
	pthread_t	*thread_id;
	pthread_attr_t	attributes;
	pthread_attr_init(&attributes);

	thread_id = (pthread_t *) malloc (sizeof(pthread_t) * num_threads);
	ARGS_FOR_THREAD *args_for_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD) * num_threads);

	pthread_barrier_init(&barrierp, NULL, num_threads);
	
	int i, j, k;

	for (i = 0; i < num_threads; i++){
		args_for_thread[i].tid = i;
		args_for_thread[i].num_threads = num_threads;
		args_for_thread[i].num_elements = U.num_rows;
		args_for_thread[i].U = U.elements;
		pthread_create(&thread_id[i], &attributes, compute_silver, (void *) &args_for_thread[i]);
	}

	for(i = 0; i < num_threads; i++){
		pthread_join(thread_id[i], NULL);
	}

	for(k = 0; k < U.num_rows; k++){
		U.elements[U.num_rows * k + k] = 1;
		for(j = k + 1; j < U.num_rows; j++){
			U.elements[U.num_rows * j + k] = 0;
		}

	}	
	
}

/* Perform Gaussian elimination in place on the U matrix */
void *compute_silver(void *args)
{
    int i, j, k;

    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) args;
    int num_elements = args_for_me->num_elements;
    float *U = args_for_me->U;
    int tid = args_for_me->tid;
    int n_t = args_for_me->num_threads;

    for (k = 0; k < num_elements; k++) {	// Rows
	/* Reduce the current row */
        for (j = k + 1 + tid; j < num_elements; j+=n_t) {   // Columns
            if (U[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
            }            
	    /* Division step */
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	
	}

	pthread_barrier_wait(&barrierp);

	/* Elimination Step */
	for (i = k + 1 + tid; i < num_elements; i+=n_t){
		for (j = k + 1; j < num_elements; j++){
			U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step */
		}
	
        }
	pthread_barrier_wait(&barrierp);
    }
    
    pthread_exit((void *) 0);
	
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}
