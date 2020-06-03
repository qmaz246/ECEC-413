/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */

#ifndef _SEPARABLE_CONVOLUTION_KERNEL_H_
#define _SEPARABLE_CONVOLUTION_KERNEL_H_



__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    /* Obtain index of thread within the overall execution grid */
 //   int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Compute the stride length = total number of threads */
  //  int stride = blockDim.x * gridDim.x;

    int i, i1;
    int j, j1, j2;
    int x, y;

    for (y = 0; y < num_rows; y++) {
        for (x = 0; x < num_cols; x++) {
            j1 = x - half_width;
            j2 = x + half_width;
            /* Clamp at the edges of the matrix */
            if (j1 < 0) 
                j1 = 0;
            if (j2 >= num_cols) 
                j2 = num_cols - 1;

            /* Obtain relative position of starting element from element being convolved */
            i1 = j1 - x; 
            
            j1 = j1 - x + half_width; /* Obtain operating width of the kernel */
            j2 = j2 - x + half_width;

            /* Convolve along row */
            result[y * num_cols + x] = 0.0f;
            for(i = i1, j = j1; j <= j2; j++, i++)
                result[y * num_cols + x] += 
                    kernel[j] * input[y * num_cols + x + i];
        }

//	thread_id += stride;
    }

    return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, int num_cols, int num_rows, int half_width)
{
    /* Obtain index of thread within the overall execution grid */
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Compute the stride length = total number of threads */
    int stride = blockDim.x * gridDim.x;


    int i, i1;
    int j, j1, j2;
    int x, y;

//    for (y = 0; y < num_rows; y++) {
    while (thread_id < (num_cols * num_rows)){
        for(x = 0; x < num_cols; x++) {
            j1 = y - half_width;
            j2 = y + half_width;
           /* Clamp at the edges of the matrix */

            if (j1 < 0) 
                j1 = 0;
            if (j2 >= num_rows) 
                j2 = num_rows - 1;

            /* Obtain relative position of starting element from element being convolved */

            i1 = j1 - y; 
            
            j1 = j1 - y + half_width; /* Obtain the operating width of the kernel.*/

            j2 = j2 - y + half_width;

            /* Convolve along column */            
            result[y * num_cols + x] = 0.0f;
            for (i = i1, j = j1; j <= j2; j++, i++)
                result[y * num_cols + x] += 
                    kernel[j] * input[y * num_cols + x + (i * num_cols)];
        }
	thread_id += stride;
    }

    return;

}

__global__ void convolve_rows_kernel_optimized()
{
    return;
}

__global__ void convolve_columns_kernel_optimized()
{
    return;
}



#endif /* _SEPARABLE_CONVOLUTION_KERNEL_H_ */ 
