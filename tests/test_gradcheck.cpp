#include "tensor.h"
#include <iostream>
#include <functional>
#include <cmath>
#include <cassert>

bool gradient_check(
    TensorPtr input,
    std::function<TensorPtr(TensorPtr)> forward_fn,
    float eps = 1e-3f,
    float tol = 1e-2
){
    input->requires_grad = true;
    input->zero_grad();
    TensorPtr loss = forward_fn(input);
    loss->backward();

    std::vector<float> analytical_grads = input->grad->data;
    bool match = true;

    for(int i = 0; i < input->numel(); i++) {
        float original_val = input->data[i];

        // +eps pass
        input->data[i] = original_val + eps;
        TensorPtr out_plus = forward_fn(input);
        float loss_plus = out_plus->data[0];

        // -eps pass
        input->data[i] = original_val - eps;
        TensorPtr out_minus = forward_fn(input);
        float loss_minus = out_minus->data[0];

        float numerical_grad = (loss_plus - loss_minus) / (2 * eps);
        float analytical_grad = analytical_grads[i];

        float diff = std::abs(numerical_grad - analytical_grad);
        float max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad));

        bool local_match = false;
        if (max_val < 1e-6f) {
            local_match = diff < tol;
        }
        else {
            float rel_err = diff / max_val;
            local_match = (rel_err < tol) || (diff < 1e-4);
        }

        if (!local_match) {
            std::cerr << "  [!] Mismatch at index " << i 
                      << " | Autograd: " << analytical_grad 
                      << " | Numerical: " << numerical_grad 
                      << " | Diff: " << diff << "\n";
            match = false;
        }
    }
    return match;

}

int main() {
    std::cout << "=== Tejas Autograd Gradient Verification ===\n";

    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{4, 4});
    a->randomize(1.0f);
    a->requires_grad = true;

    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{4, 4});
    b->randomize(1.0f);
    b->requires_grad = false;

    bool relu_pass = gradient_check(a, [&](TensorPtr x){ return sum(relu(x)); });
    std::cout << (relu_pass ? "[PASS]" : "[FAIL]") << " ReLU\n";

    bool add_pass = gradient_check(a, [&](TensorPtr x){ return sum(add(x, b)); });
    std::cout << (add_pass ? "[PASS]" : "[FAIL]") << " Add\n";

    bool mul_pass = gradient_check(a, [&](TensorPtr x){ return sum(multiply(x, b)); });
    std::cout << (mul_pass ? "[PASS]" : "[FAIL]") << " Multiply\n";

    bool matmul_pass = gradient_check(a, [&](TensorPtr x){ return sum(matmul(x, b)); });
    std::cout << (matmul_pass ? "[PASS]" : "[FAIL]") << " Matmul\n";

    if(relu_pass && add_pass && mul_pass && matmul_pass) {
        std::cout << "\nSUCCESS: Autograd calculus perfectly matches numerical approximations!\n";
    } else {
        std::cout << "\nFAILURE: Autograd engine contains calculus errors.\n";
        assert(false);
    }

    return 0;

}