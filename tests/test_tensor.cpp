#include<bits/stdc++.h>
#include "tensor.h"

bool approx_equal(float a, float b, float eps = 1e-4f){
    return std::abs(a - b) < eps;
}

void test_tensor_basics(){
    std::cout << "=== tensor basics ===\n";
    Tensor t({2, 3});
    t.at({0,0}) = 1.0f; t.at({0,1}) = 2.0f; t.at({0,2}) = 3.0f;
    t.at({1,0}) = 4.0f; t.at({1,1}) = 5.0f; t.at({1,2}) = 6.0f;
    t.print();
    std::cout << "strides: [" << t.strides[0] << ", " << t.strides[1] << "]\n";
    std::cout << (t.at({1,2}) == 6.0f ? "PASS" : "FAIL") << " at() read\n";
    t.at({1,2}) = 99.0f;
    std::cout << (t.at({1,2}) == 99.0f ? "PASS" : "FAIL") << " at() write\n\n";
}

void test_matmul_correctness(){
    std::cout << "=== matmul correctness ===\n";
    // 2x3 @ 3x2 = 2x2, known answer [[22,28],[49,64]]
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({3,2}, {1,2,3,4,5,6});
    Tensor c_naive = matmul(a, b);
    Tensor c_tiled = matmul_tiled(a, b);
    std::vector<float> expected = {22,28,49,64};
    bool naive_ok = true, tiled_ok = true;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++){
            if(!approx_equal(c_naive.at({i,j}), expected[i*2+j])) naive_ok = false;
            if(!approx_equal(c_tiled.at({i,j}), expected[i*2+j])) tiled_ok = false;
        }
    std::cout << (naive_ok ? "PASS" : "FAIL") << " naive matmul\n";
    std::cout << (tiled_ok ? "PASS" : "FAIL") << " tiled matmul\n\n";
}

void test_matmul_non_multiple_of_tile(){
    std::cout << "=== non-tile-multiple dimensions ===\n";
    // 5x7 @ 7x3 — not a multiple of tile size 4
    int M=5, K=7, N=3;
    std::vector<float> a_data, b_data;
    for(int i=0;i<M*K;i++) a_data.push_back((float)(i+1));
    for(int i=0;i<K*N;i++) b_data.push_back((float)(i+1));
    Tensor a({M,K}, a_data);
    Tensor b({K,N}, b_data);
    Tensor c_naive = matmul(a, b);
    Tensor c_tiled = matmul_tiled(a, b);
    bool ok = true;
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            if(!approx_equal(c_naive.at({i,j}), c_tiled.at({i,j}))) ok = false;
    std::cout << (ok ? "PASS" : "FAIL") << " tiled matches naive for non-tile-multiple dims\n\n";
}

void test_matmul_large(){
    std::cout << "=== large matrix naive vs tiled ===\n";
    int M=64, K=64, N=64;
    std::vector<float> a_data, b_data;
    for(int i=0;i<M*K;i++) a_data.push_back((float)(rand()%10));
    for(int i=0;i<K*N;i++) b_data.push_back((float)(rand()%10));
    Tensor a({M,K}, a_data);
    Tensor b({K,N}, b_data);
    Tensor c_naive = matmul(a, b);
    Tensor c_tiled = matmul_tiled(a, b);
    bool ok = true;
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            if(!approx_equal(c_naive.at({i,j}), c_tiled.at({i,j}))) ok = false;
    std::cout << (ok ? "PASS" : "FAIL") << " 64x64 tiled matches naive\n\n";
}
void test_elementwise(){
    std::cout << "=== elementwise ops ===\n";
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({2,3}, {6,5,4,3,2,1});

    Tensor c = add(a, b);
    bool add_ok = true;
    for(int i = 0; i < c.numel(); i++)
        if(!approx_equal(c.data[i], 7.0f)) add_ok = false;
    std::cout << (add_ok ? "PASS" : "FAIL") << " add\n";

    Tensor d = multiply(a, b);
    std::vector<float> expected_mul = {6,10,12,12,10,6};
    bool mul_ok = true;
    for(int i = 0; i < d.numel(); i++)
        if(!approx_equal(d.data[i], expected_mul[i])) mul_ok = false;
    std::cout << (mul_ok ? "PASS" : "FAIL") << " multiply\n";

    Tensor e({2,3}, {-3,-1,0,1,2,3});
    Tensor f = relu(e);
    std::vector<float> expected_relu = {0,0,0,1,2,3};
    bool relu_ok = true;
    for(int i = 0; i < f.numel(); i++)
        if(!approx_equal(f.data[i], expected_relu[i])) relu_ok = false;
    std::cout << (relu_ok ? "PASS" : "FAIL") << " relu\n\n";
}
int main(){
    test_tensor_basics();
    test_matmul_correctness();
    test_matmul_non_multiple_of_tile();
    test_matmul_large();
    test_elementwise();
    return 0;
}