/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
__global__ void delta3_kernel(float *delta3, float *train_label, float *outa, int classes);
__global__ void delta2_kernel(float *delta2, float *delta3, float *W3, float *h2a, int classes);
__global__ void delta1_kernel(float *delta1, float *delta2, float *W2, float *h1a);
