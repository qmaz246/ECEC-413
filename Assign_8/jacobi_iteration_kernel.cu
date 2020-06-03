#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(matrix_t A, matrix_t B, matrix_t X_in, matrix_t X_out)
{
    int i = threadIdx.x;
    int j = 0;
    int sum = 0;

    for(j = 0; j < num_elements; j++){
        if (i != j)
            sum += A.elements[i * num_elements + j] * X_in.elements[j];
    }

    X_out.elements[i] = (B.elements[i] - sum)/A.elements[i * num_elements + i];

    return;
}

__global__ void jacobi_iteration_kernel_optimized(matrix_t A, matrix_t B, matrix_t X_in, matrix_t X_out)
{
    return;
}

