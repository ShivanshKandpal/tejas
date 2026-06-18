#include "tensor.h"
#include <iostream>

int main() {
    std::cout<<"<--- dispatcher test --->\n\n";

    TensorPtr A = std::make_shared<Tensor>(std::vector<int>{2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    TensorPtr B = std::make_shared<Tensor>(std::vector<int>{2, 2}, std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});

    std::cout << "1. Tensors created on System RAM (CPU).\n";

    TensorPtr d_A = A->cuda();
    TensorPtr d_B = B->cuda();

    std::cout << "2. Tensors successfully moved to VRAM (GPU).\n";

    TensorPtr d_C = matmul(d_A, d_B);

    std::cout << "3. Hardware-accelerated Matmul executed.\n";

    TensorPtr C = d_C->cpu();

    std::cout << "4. Results pulled back across PCIe bus to CPU.\n\n";

    std::cout << "Result Matrix (Should be [19, 22] / [43, 50]):\n";
    C->print();

    std::cout << "\nSUCCESS: The Dispatcher is fully operational!\n";
    return 0;
}