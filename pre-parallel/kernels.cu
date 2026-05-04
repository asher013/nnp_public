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

// ----------------FORWARD PASS KERNELS--------------------

// Each thread computes one neuron's pre-activation and activation value.

__global__ void forward_layer_relu(
    float *input,    // input vector
    float *W,        // input matrix
    float *b,        // (out_size,)
    float *pre,      
    float *act,      
    int in_size,
    int out_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= out_size) return; // Guards against out-of-bounds indexing

    float sum = b[row];
    for (int i = 0; i < in_size; i++) {
        sum += input[i] * W[i * out_size + row];
    }

    pre[row] = sum;
    act[row] = sum > 0.0f ? sum : 0.0f;   // This will store the output values after the ReLu activation function is applied
}

// linear activation kernel for final layer

__global__ void forward_layer_linear(float *input, float *W, float *b, float *out, int in_size, int out_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= out_size) return;

    float sum = b[row];
    for (int i = 0; i < in_size; i++) {
        sum += input[i] * W[i * out_size + row];
    }

    out[row] = sum;
}

// --------------- WEIGHT UPDATE KERNELS ---------------------
__global__ void weight_updates(float *W, float *input, float *delta, int in_size, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= in_size * out_size) return;

    // Recover 2d indices

    int j = idx / out_size;
    int k = idx % out_size;

    W[idx] += LR * delta[k] * input[j];

}