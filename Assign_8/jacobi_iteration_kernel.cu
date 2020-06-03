#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float * B, float *X_in, float *X_out, int *num_elements, double *ssd)
{
    int thread = threadIdx.x;
    int block = blockIdx.x;
    int i = (block * THREAD_BLOCK_SIZE) + thread; 
    int j = 0;
    double sum = 0;
    int num_el = *num_elements;

    for(j = 0; j < num_el; j++){
        if (i != j)
            sum += A[i * num_el + j] * X_in[j];
    }

    X_out[i] = (B[i] - sum)/A[i * num_el + i];

    ssd[i] = (X_in[i] - X_out[i]) * (X_in[i] - X_out[i]);

    return;
}

__global__ void jacobi_iteration_kernel_optimized(matrix_t A, matrix_t B, matrix_t X_in, matrix_t X_out)
{
    return;
}

