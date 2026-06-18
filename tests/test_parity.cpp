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
    std::cout<<" Tejas Elementwise Parity Check \n";
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

    std::cout << "\nSUCCESS: All GPU operations perfectly match CPU baseline!\n";
    return 0;
}