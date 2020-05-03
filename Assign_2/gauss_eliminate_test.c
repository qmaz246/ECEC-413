/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): Edward Mazzilli & Clayton DeGruchy
 * Date: May 06, 2020
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#define _GNU_SOURCE


#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50


/* Function prototypes */
extern int compute_gold(float *, int);
void *compute_silver(void *args);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
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

    Matrix A;			      /* Input matrix */
    Matrix U_reference;		      /* Upper triangular matrix computed by reference code */
    Matrix U_mt;		      /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                     /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);       /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);  /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);   /* Allocate space for multi-threaded result */
    //print_matrix(A); 
    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
    //print_matrix(U_reference); 
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* Perform Gaussian elimination using pthreads at 4, 8, and 16 threads
     * The resulting upper triangular matrix should be returned in U_mt */

    for(int k = 4; k<=16; k=k*2){
 	fprintf(stderr, "\nPerforming gaussian elimination using %i pthreads\n", k);
	gettimeofday(&start, NULL);
	gauss_eliminate_using_pthreads(U_mt, k);
    	gettimeofday(&stop, NULL);

	/* Check if pthread result matches reference solution within specified tolerance */

	fprintf(stderr, "\nChecking results\n");
   	int size = matrix_size * matrix_size;
   	int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    	fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");
	fprintf(stderr, "%d threads run time = %0.2f s\n", k, (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    }

/*
	gettimeofday(&start, NULL);
	gauss_eliminate_using_pthreads(U_mt, 4);
    	gettimeofday(&stop, NULL);
	print_matrix(U_mt);
	fprintf(stderr, "\nChecking results\n");
   	int size = matrix_size * matrix_size;
   	int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    	fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");
*/
    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
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

void print_matrix(const Matrix n)
{
	int i, j;
	for (i = 0; i < n.num_rows; i++){
		for (j = 0; j < n.num_rows; j++){
			printf("%.2f ", n.elements[n.num_rows * i + j]);
		}
		printf("\n");
	}
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
