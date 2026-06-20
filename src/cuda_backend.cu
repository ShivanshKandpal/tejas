#include "backend.h"
#include <cuda_runtime.h>
#include <math_constants.h> //cudart_inf_f
#include <cassert>
#include <iostream>

inline int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;
    int num_phases = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int ph = 0; ph < num_phases; ph++){
        if(row < M && (ph * TILE_SIZE + tx) < K){
            As[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if((ph * TILE_SIZE + ty) < K && col < N){
            Bs[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; k++){
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

TensorPtr cuda_matmul_tiled_wrapper(const TensorPtr& a, const TensorPtr& b) {
    assert(a->shape[1] == b->shape[0]);
    assert(a->device == Device::CUDA && b->device == Device::CUDA);

    int M = a->shape[0];
    int N = b->shape[1];
    int K = a->shape[1];

    TensorPtr result = std::make_shared<Tensor>(std::vector<int>{M, N});
    result->device = Device::CUDA;

    int total_bytes = M * N * sizeof(float);
    cudaMalloc(&result->gpu_data, total_bytes);

    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((N + TILE_SIZE - 1)/TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<num_blocks, threads_per_block>>>(a->gpu_data, b->gpu_data, result->gpu_data, M, N, K);

    cudaDeviceSynchronize();

    return result;
}

//elementwise kernels

__global__ void add_kernel(const float* a, const float* b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] + b[i];
}

__global__ void multiply_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) c[i] = a[i] * b[i];
}

__global__ void relu_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) output[i] = input[i] > 0.0f ? input[i] : 0.0f;

}


//elementwise wrappers

TensorPtr cuda_add_wrapper(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    result->device = Device::CUDA;
    cudaMalloc(&result->gpu_data, a->numel() * sizeof(float));

    int threads = 256;
    int blocks = (a->numel() + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a->gpu_data, b->gpu_data, result->gpu_data, a->numel());
    cudaDeviceSynchronize();
    return result;
}

TensorPtr cuda_multiply_wrapper(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    result->device = Device::CUDA;
    cudaMalloc(&result->gpu_data, a->numel() * sizeof(float));

    int threads = 256;
    int blocks = (a->numel() + threads - 1) / threads;
    multiply_kernel<<<blocks, threads>>>(a->gpu_data, b->gpu_data, result->gpu_data, a->numel());
    cudaDeviceSynchronize();

    return result;
}

TensorPtr cuda_relu_wrapper(const TensorPtr& a) {
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    result->device = Device::CUDA;
    cudaMalloc(&result->gpu_data, a->numel() * sizeof(float));

    int threads = 256;
    int blocks = (a->numel() + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(a->gpu_data, result->gpu_data, a->numel());
    cudaDeviceSynchronize();

    return result;

}


// softmax kernel

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * cols + col;

    if(col < cols){
        sdata[col] = input[idx];
    }
    else {
        sdata[col] = -CUDART_INF_F;
    }  
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(col < stride) {
            sdata[col] = fmaxf(sdata[col], sdata[col + stride]);
        }
        __syncthreads();
    }

    float row_max = sdata[0];
    __syncthreads();

    float exp_val = 0.0f;
    if(col < cols) {
        exp_val = expf(input[idx] - row_max);
        sdata[col] = exp_val;
    }
    else {
        sdata[col] = 0.0f;
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(col < stride) {
            sdata[col] += sdata[col + stride];
        }
        __syncthreads();
    }

    float row_sum = sdata[0];

    if(col < cols) {
        output[idx] = exp_val / row_sum;
    }

}

// softmax wrapper

TensorPtr cuda_softmax_wrapper(const TensorPtr& a) {
    assert(a->shape.size() <= 2);

    int rows = (a->shape.size() > 1) ? a->shape[0] : 1;
    int cols = (a->shape.size() > 1) ? a->shape[1] : a->shape[0];

    assert(cols <= 1024);   

    int block_size = next_power_of_2(cols);

    TensorPtr result = std::make_shared<Tensor>(a->shape);
    result->device = Device::CUDA;
    cudaMalloc(&result->gpu_data, a->numel() * sizeof(float));

    softmax_kernel<<<rows, block_size, block_size * sizeof(float)>>>(
        a->gpu_data, result->gpu_data, rows, cols
    );
    cudaDeviceSynchronize();

    return result;
}
