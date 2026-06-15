#include <iostream>
#include<chrono>
#include<vector>
__global__ void add_kernel(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n)
        c[i] = a[i] + b[i];
}

int main(){
    int n = 1000000;
    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for(int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int i = 0;i<n;i++){
        h_c_cpu[i] = h_a[i]+h_b[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli> (end_cpu - start_cpu).count();

    float *d_a, *d_b, *d_c;
    auto start_total = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    auto start_kernel = std::chrono::high_resolution_clock::now();
    add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_c_gpu.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_total = std::chrono::high_resolution_clock::now();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    double ms_gpu_total = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    double ms_gpu_kernel = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    bool match = true;
    for(int i = 0; i < n; i++){
        if(std::abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i] << ", GPU=" << h_c_gpu[i] << "\n";
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