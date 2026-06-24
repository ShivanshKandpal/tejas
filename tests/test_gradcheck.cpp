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

        input->data[i] = original_val;

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

    bool softmax_pass = gradient_check(a, [&](TensorPtr x){ 
    return sum(multiply(softmax(x), b)); });
    std::cout << (softmax_pass ? "[PASS]" : "[FAIL]") << " Softmax\n";

    bool add_pass = gradient_check(a, [&](TensorPtr x){ return sum(add(x, b)); });
    std::cout << (add_pass ? "[PASS]" : "[FAIL]") << " Add\n";

    bool mul_pass = gradient_check(a, [&](TensorPtr x){ return sum(multiply(x, b)); });
    std::cout << (mul_pass ? "[PASS]" : "[FAIL]") << " Multiply\n";

    bool matmul_pass = gradient_check(a, [&](TensorPtr x){ return sum(matmul(x, b)); });
    std::cout << (matmul_pass ? "[PASS]" : "[FAIL]") << " Matmul\n";
    
    bool transpose_pass = gradient_check(a, [&](TensorPtr x){ return sum(transpose(x)); });

    std::cout << (transpose_pass ? "[PASS]" : "[FAIL]") << " Transpose\n";


    // attention gradient check
    int seq_len = 4;
    int d_model = 8;
    int d_k     = 4;
    TensorPtr W_q = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_q->randomize();
    TensorPtr W_k = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_k->randomize();
    TensorPtr W_v = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_v->randomize();

    TensorPtr b_q = std::make_shared<Tensor>(std::vector<int>{1, d_k}); b_q->randomize();
    TensorPtr b_k = std::make_shared<Tensor>(std::vector<int>{1, d_k}); b_k->randomize();
    TensorPtr b_v = std::make_shared<Tensor>(std::vector<int>{1, d_k}); b_v->randomize();

    TensorPtr X = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model}); X->randomize();
    X->requires_grad = true;

    auto attn_forward = [&](TensorPtr input){
        TensorPtr Q = add(matmul(input, W_q), b_q);
        TensorPtr K = add(matmul(input, W_k), b_k);
        TensorPtr V = add(matmul(input, W_v), b_v);

        TensorPtr scores = matmul(Q, transpose(K));
        scores = scale(scores, 1.0f / std::sqrt(d_k));

        TensorPtr attn = softmax(scores);

        return sum(matmul(attn, V));
    };

    bool attn_pass = gradient_check(X, attn_forward);
    std::cout << (attn_pass ? "[PASS]" : "[FAIL]") << " Attention Block\n";

    TensorPtr X_mat = std::make_shared<Tensor>(std::vector<int>{4, 4});
    X_mat->randomize(1.0f);
    X_mat->requires_grad = true;

    TensorPtr b_vec = std::make_shared<Tensor>(std::vector<int>{1, 4});
    b_vec->randomize(1.0f);
    b_vec->requires_grad = true;

    bool broad_pass_X = gradient_check(X_mat, [&](TensorPtr x) { return sum(add(x, b_vec)); });

    bool broad_pass_b = gradient_check(b_vec, [&](TensorPtr x) { return sum(add(X_mat, x)); });

    bool broad_pass = broad_pass_b && broad_pass_X;
    std::cout << (broad_pass ? "[PASS]" : "[FAIL]") << " Broadcasting Add\n";

    if(relu_pass && add_pass && mul_pass && matmul_pass && softmax_pass && transpose_pass && attn_pass && broad_pass) {
        std::cout << "\nSUCCESS: Autograd calculus perfectly matches numerical approximations!\n";
    } else {
        std::cout << "\nFAILURE: Autograd engine contains calculus errors.\n";
        assert(false);
    }
    
    return 0;

}