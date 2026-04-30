/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

#include "config.h"
#include "kernels.h"
#include "nnp.h"


// -------------Backprop kernels----------------
__global__ void delta3_kernel(float *delta3, float *train_label, float *outa, int classes) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Get thread's ID
    if (k < classes) { 
        delta3[k] = train_label[k] - outa[k];
    }
}

// delta 2 relies on delta3
__global__ void delta2_kernel(float *delta2, float *delta3, float *W3 , float *h2a, int classes){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err = 0;
    if (j < H2) {
        for (int k = 0; k < classes; k++)
            err += delta3[k] * W3[j * classes + k];
        delta2[j] = err * drelu(h2a[j]);
    }
} 

// delta 1 relies on delta 2
__global__ void delta1_kernel(float *delta1, float *delta2, float *W2 , float *h1a){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float err = 0;
    if (j < H1) {
        for (int k = 0; k < H2; k++)
            err += delta2[k] * W2[j * H2 + k];
        delta1[j] = err * drelu(h1a[j]);
    }
}