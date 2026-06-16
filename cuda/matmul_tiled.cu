#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>

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
        } 
        else{
            As[ty][tx] = 0.0f;
        }

        if((ph * TILE_SIZE + ty) < K && col < N){
            Bs[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];
        }
        else{
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

int main(){
    int M = 512, N = 512, K = 512;
    int size_A = M * K, size_B = K * N, size_C = M * N;

    srand(time(NULL));

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_cpu(size_C, 0.0f);
    std::vector<float> h_C_gpu(size_C, 0.0f);

    for (int i = 0; i < size_A; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < size_B; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    int tile = 32; 
    for(int tile_i=0; tile_i<M; tile_i+=tile){
        for(int tile_j=0; tile_j<N; tile_j+=tile){
            for(int tile_k=0; tile_k<K; tile_k+=tile){
                int i_end = std::min(tile_i+tile, M);
                int j_end = std::min(tile_j+tile, N);
                int k_end = std::min(tile_k+tile, K);
                for(int i = tile_i; i < i_end; i++){
                    for(int k = tile_k; k < k_end; k++){
                        float a_ik = h_A[i * K + k]; 
                        for(int j = tile_j; j < j_end; j++){
                            h_C_cpu[i * N + j] += a_ik * h_B[k * N + j]; 
                        }
                    }
                }
            }
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float *d_A, *d_B, *d_C;
    auto start_total = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    auto start_kernel = std::chrono::high_resolution_clock::now();
    matmul_tiled_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_total = std::chrono::high_resolution_clock::now();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    double ms_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    double ms_gpu_kernel = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    double ms_gpu_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    bool match = true;
    for (int i = 0; i < size_C; i++) {
        float diff = std::abs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff / std::abs(h_C_cpu[i]) > 1e-2) { 
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i] << ", GPU=" << h_C_gpu[i] << "\n";
            break;
        }
    }

    std::cout << "\n=== Tiled Matmul Benchmark (512x512) ===\n";
    std::cout << "Parity Check: " << (match ? "PASS" : "FAIL") << "\n";
    std::cout << "CPU Time:       " << ms_cpu << " ms\n";
    std::cout << "GPU Math Time:  " << ms_gpu_kernel << " ms\n";
    std::cout << "Speedup vs CPU: " << ms_cpu / ms_gpu_kernel << "x\n";
}