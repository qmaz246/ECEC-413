/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
//#define DEBUG

/* Structure to pass arguments into various threads */
typedef struct args_for_thread
{
	int		tid;
	int		num_threads;
	int		num_elements;
	matrix_t 	A;
	matrix_t 	B;
	matrix_t 	mt_sol_x;
	matrix_t	new_x;
        pthread_barrier_t *barrier;
	int		rows;
	int		start;
	int		stop;
	int		max_iter;
} ARGS_FOR_THREAD;

int main(int argc, char **argv) 
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    struct timeval start, stop;	

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time Gold = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n\n");
    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B, 4, matrix_size, max_iter);
    gettimeofday(&stop, NULL); 
    fprintf(stderr, "Execution time 4 Threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));   
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);    

    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B, 8, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time 8 Threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B, 16, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time 16 Threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

    gettimeofday(&start, NULL);
    compute_using_pthreads(A, mt_solution_x, B, 32, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time 32 Threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    mt_solution_x = allocate_matrix(matrix_size, 1, 0);

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads, int matrix_size, int max_iter)
{
	pthread_t	*thread_id;
	pthread_attr_t	attributes;
	pthread_attr_init(&attributes);

	thread_id = (pthread_t *) malloc (sizeof(pthread_t) * num_threads);
	ARGS_FOR_THREAD *args_for_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD) * num_threads);

	pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
	pthread_barrier_init(barrier,NULL,num_threads);

        /* Allocate n x 1 matrix to hold iteration values.*/
        matrix_t new_x = allocate_matrix(matrix_size, 1, 0);

        int i;

        /* Initialize current jacobi solution. */
        for (i = 0; i <= matrix_size; i++)
            mt_sol_x.elements[i] = B.elements[i];

	int rows_per_thread = matrix_size / num_threads;

	for (i = 0; i < num_threads; i++){
		args_for_thread[i].tid = i;
		args_for_thread[i].num_threads = num_threads;
		args_for_thread[i].num_elements = matrix_size;
		args_for_thread[i].A = A;
		args_for_thread[i].B = B;
		args_for_thread[i].mt_sol_x = mt_sol_x;
		args_for_thread[i].start = i * rows_per_thread;  
		args_for_thread[i].stop = (i * rows_per_thread) + rows_per_thread -1;
		args_for_thread[i].max_iter = max_iter;
		args_for_thread[i].new_x = new_x;
                args_for_thread[i].barrier = barrier;
		pthread_create(&thread_id[i], &attributes, compute_silver, (void *) &args_for_thread[i]);
	}

	for(i = 0; i < num_threads; i++){
		pthread_join(thread_id[i], NULL);
	}
	
}

// Perform Jacobi
void *compute_silver(void *args)
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *) args;
    int num_elements = args_for_me->num_elements;
    int tid = args_for_me->tid;
    matrix_t A = args_for_me->A;
    matrix_t B = args_for_me->B;
    matrix_t mt_sol_x = args_for_me->mt_sol_x;
    matrix_t new_x = args_for_me->new_x;
    int start = args_for_me->start;
    int stop = args_for_me->stop;
    int max_iter = args_for_me->max_iter;
    pthread_barrier_t *barrier = args_for_me->barrier;

    int i, j;

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;

    while (!done) {
        for (i = start; i <= stop; i++) {
            double sum = 0.0;
            for (j = 0; j < num_elements; j++) {
                if (i != j)
                    sum += A.elements[i * num_elements + j] * mt_sol_x.elements[j];
            }

            /* Update values for the unkowns for the current row. */
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_elements + i];
        }

        pthread_barrier_wait(barrier);

        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
        for (i = 0; i < num_elements; i++) {
            ssd += (new_x.elements[i] - mt_sol_x.elements[i]) * (new_x.elements[i] - mt_sol_x.elements[i]);
        }

        pthread_barrier_wait(barrier);

	for (i = start; i <= stop; i++) {
            mt_sol_x.elements[i] = new_x.elements[i];
        }

	num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */

	if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;
    }

    if (tid == 0){
    	printf("Num Iter: %d\n", num_iter);
    	printf("MSE: %f\n", mse);
    }

    pthread_exit((void *) 0);
}


/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
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

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



