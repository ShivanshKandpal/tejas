#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>

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

void run_benchmark(int size) {
    int M = size, N = size, K = size;
    int size_A = M * K, size_B = K * N, size_C = M * N;
    int num_runs = 10;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_cpu(size_C, 0.0f);
    
    for (int i = 0; i < size_A; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < size_B; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;


    double total_ms_cpu = 0.0;
    for (int r = 0; r < num_runs; r++) {
        
        std::fill(h_C_cpu.begin(), h_C_cpu.end(), 0.0f);
        
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < M; i++){
            for(int k = 0; k < K; k++){
                float a_ik = h_A[i * K + k];
                for(int j = 0; j < N; j++){
                    h_C_cpu[i * N + j] += a_ik * h_B[k * N + j]; 
                }
            }
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        total_ms_cpu += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    }
    double avg_ms_cpu = total_ms_cpu / num_runs;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    double total_ms_gpu = 0.0;
    for (int r = 0; r < num_runs; r++) {
        auto start_gpu = std::chrono::high_resolution_clock::now();
        matmul_tiled_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        total_ms_gpu += std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
    }
    double avg_ms_gpu = total_ms_gpu / num_runs;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    double speedup = avg_ms_cpu / avg_ms_gpu;

    std::cout << "| " << size << " | " 
              << std::fixed << std::setprecision(2) << avg_ms_cpu << " | " 
              << avg_ms_gpu << " | " 
              << speedup << "x |\n";
}

int main(){
    srand(time(NULL));
    std::cout << "Compiling Markdown Benchmark Table...\n\n";
    std::cout << "| Size | Optimized CPU (ms) | Tiled GPU Kernel (ms) | Speedup |\n";
    std::cout << "| ----- | ----- | ----- | ----- |\n";
    
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048};
    for(int s : sizes) {
        run_benchmark(s);
    }
    
    std::cout << "\n";
    return 0;
}