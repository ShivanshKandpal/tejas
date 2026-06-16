#include<iostream>
#include<chrono>
#include<vector>
#include<cmath>
#include<ctime>
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < M && col < N){
        float sum = 0.0f;
        for(int k = 0;k<K;k++){
            sum += A[row*K + k] * B[k*N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(){
    srand(time(NULL));
    int M = 512, N = 512, K = 512;
    int size_A = M*K, size_B = K*N, size_C = M*N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_cpu(size_C, 0.0f);
    std::vector<float> h_C_gpu(size_C, 0.0f);

    for(int i = 0; i < size_A; i++){
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    for(int i = 0; i < size_B; i++){
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < K; k++){
                sum += h_A[i * K + k] * h_B[k*N + j];
            }
            h_C_cpu[i * N + j] = sum; 
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
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((N+15)/16, (M+15)/16);

    auto start_kernel = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);
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
    for(int i = 0; i < size_C; i++){
        if(std::abs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5){
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_C_cpu[i] << ", GPU=" << h_C_gpu[i] << "\n";
            break;
        }
    }
    std::cout << "CPU time: " << ms_cpu << " ms\n";
    std::cout << "GPU total time: " << ms_gpu_total << " ms\n";
    std::cout << "GPU kernel time: " << ms_gpu_kernel << " ms\n";
    std::cout << "Results match: " << (match ? "YES" : "NO") << "\n";
    std::cout << "Speedup (total): " << ms_cpu / ms_gpu_total << "x\n";
    std::cout << "Speedup (kernel only): " << ms_cpu / ms_gpu_kernel << "x\n";
    return 0;
}