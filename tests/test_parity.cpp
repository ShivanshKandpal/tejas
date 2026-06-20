#include "tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void assert_tensors_equal(TensorPtr t1, TensorPtr t2) {
    assert(t1->numel() == t2->numel());
    for(int i = 0; i < t1->numel(); i++) {
        if(std::abs(t1->data[i] - t2->data[i]) > 1e-5) {
            std::cerr<<" Mismatch at index "<<i<<" : CPU "<<t1->data[i]<<" != GPU "<<t2->data[i]<<"\n";
            assert(false);
        }
    }
}

int main() {
    std::cout << "--- Tejas Hardware Parity Check ---\n";
    TensorPtr A = std::make_shared<Tensor>(std::vector<int>{2, 2}, std::vector<float>{-1.0f, 2.0f, -3.0f, 4.0f});
    TensorPtr B = std::make_shared<Tensor>(std::vector<int>{2, 2}, std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});

    TensorPtr d_A = A->cuda();
    TensorPtr d_B = B->cuda();

    TensorPtr C_add = add(A, B);
    TensorPtr d_C_add = add(d_A, d_B)->cpu();
    assert_tensors_equal(C_add, d_C_add);
    std::cout<<"[PASS] Add (CPU vs GPU)\n";

    TensorPtr C_mul = multiply(A, B);
    TensorPtr d_C_mul = multiply(d_A, d_B)->cpu();
    assert_tensors_equal(C_mul, d_C_mul);
    std::cout<<"[PASS] Multiply (CPU vs GPU)\n";

    TensorPtr C_relu = relu(A);
    TensorPtr d_C_relu = relu(d_A)->cpu();
    assert_tensors_equal(C_relu, d_C_relu);
    std::cout<<"[PASS] ReLU (CPU vs GPU)\n";

    TensorPtr A_big = std::make_shared<Tensor>(std::vector<int>{64, 64});
    TensorPtr B_big = std::make_shared<Tensor>(std::vector<int>{64, 64});
    A_big->randomize(); B_big->randomize();

    TensorPtr d_A_big = A_big->cuda();
    TensorPtr d_B_big = B_big->cuda();

    TensorPtr C_cpu = matmul(A_big, B_big);
    TensorPtr C_gpu = matmul(d_A_big, d_B_big)->cpu();

    bool matmul_match = true;
    for(int i = 0; i < C_cpu->numel(); i++) {
        float rel_err = std::abs(C_cpu->data[i] - C_gpu->data[i]) / (std::abs(C_cpu->data[i] + 1e-6f));

        if(rel_err > 1e-3) {
            std::cerr << "Matmul mismatch at index " << i
            << ": CPU " << C_cpu->data[i]
            << " != GPU " << C_gpu->data[i]
            << " (Rel Err: " << rel_err << ")\n";
            matmul_match = false;
            break;
        }
    }

    if (matmul_match) {
        std::cout << "[PASS] Matmul (CPU vs GPU)\n";
    } else {
        std::cout << "[FAIL] Matmul (CPU vs GPU)\n";
        assert(false);
    }

    TensorPtr A_sm = std::make_shared<Tensor>(std::vector<int>{7, 13});
    for (int i = 0; i < A_sm->numel(); i++) A_sm->data[i] = (float)(rand() % 100) / 10.0f; 

    TensorPtr d_A_sm = A_sm->cuda();

    TensorPtr sm_cpu = softmax(A_sm);
    TensorPtr sm_gpu = softmax(d_A_sm)->cpu();

    bool sm_match = true;
    for(int i = 0; i < sm_cpu->numel(); i++) {
        float rel_err = std::abs(sm_cpu->data[i] - sm_gpu->data[i]) / 
                        (std::abs(sm_cpu->data[i]) + 1e-6f);
        
        if(rel_err > 1e-3f) { 
            std::cerr << "Softmax mismatch at index " << i 
                      << ": CPU " << sm_cpu->data[i] 
                      << " != GPU " << sm_gpu->data[i] << "\n";
            sm_match = false; 
            break; 
        }
    }
    
    if (sm_match) {
        std::cout << "[PASS] Softmax (CPU vs GPU)\n";
    } else {
        std::cout << "[FAIL] Softmax (CPU vs GPU)\n";
        assert(false);
    }
    
    std::cout << "\nSUCCESS: All GPU operations perfectly match CPU baseline!\n";
    return 0;
}