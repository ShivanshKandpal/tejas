#include "backend.h"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

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