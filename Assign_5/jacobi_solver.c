/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 29, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -fopenmp -std=c99 -Wall -O3 -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"
#include <omp.h>

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

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
    printf("\n");
	
    /* Compute the Jacobi solution using openMP. 
     * Solution is returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using omp on 4 threads\n");
    gettimeofday(&start, NULL);
	compute_using_omp(A, mt_solution_x, B, 4, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time on 4 threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    printf("\n");    

    fprintf(stderr, "\nPerforming Jacobi iteration using omp on 8 threads\n");
    gettimeofday(&start, NULL);
        compute_using_omp(A, mt_solution_x, B, 8, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time on 8 threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    printf("\n");

    fprintf(stderr, "\nPerforming Jacobi iteration using omp on 16 threads\n");
    gettimeofday(&start, NULL);
        compute_using_omp(A, mt_solution_x, B, 16, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Execution time on 16 threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    printf("\n");

    fprintf(stderr, "\nPerforming Jacobi iteration using omp on 32 threads\n");
    gettimeofday(&start, NULL);
        compute_using_omp(A, mt_solution_x, B, 32, matrix_size, max_iter);
    gettimeofday(&stop, NULL);
fprintf(stderr, "Execution time on 32 threads = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    printf("\n");

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using openMP. 
 * Result must be placed in mt_sol_x. */
void compute_using_omp(const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads, int matrix_size, int max_iter)
{
	int i, j;
    	int done = 0;
    	double ssd, mse, sum;
    	int num_iter = 0;
	int num_elements = matrix_size; 

	matrix_t new_x;
	new_x = allocate_matrix(matrix_size, 1, 0);

        /* Initialize current jacobi solution. */
        for (i = 0; i < matrix_size; i++)
            mt_sol_x.elements[i] = B.elements[i];

	omp_set_num_threads(num_threads);

	while (done != 1){

	#pragma omp parallel for default(none) shared(num_elements, mt_sol_x, new_x) private(i, j, sum)
	for (i = 0; i < num_elements; i++) {
		sum = 0.0;
		for (j = 0; j < num_elements; j++){
			if (i != j)
                    		sum += A.elements[i * num_elements + j] * mt_sol_x.elements[j];
		}
		/* Update values for the unkowns for the current row. */
		new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_elements + i];		
	}


        /* Check for convergence and update the unknowns. */
        ssd = 0.0; 
        for (i = 0; i < num_elements; i++) {
            ssd += (mt_sol_x.elements[i] - new_x.elements[i]) * (mt_sol_x.elements[i] - new_x.elements[i]);
        }


	#pragma omp parallel for default(none) shared(mt_sol_x, new_x, num_elements) private(i)
	for (i = 0; i < num_elements; i++) {
        	mt_sol_x.elements[i] = new_x.elements[i];
        }

	num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */

	if (mse <= THRESHOLD)
            done = 1;

	}

	printf("Convergence achieved after %d iterations\n", num_iter);
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
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
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



