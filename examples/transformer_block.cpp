#include "tensor.h"
#include "nn.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "=== Tejas Transformer Block ===\n\n";

    int seq_len = 4;
    int d_model = 8;
    int d_k     = 4;
    int d_ff    = 32;

    tejas::nn::TransformerBlock block(d_model, d_k, d_ff);

    // --- architecture summary ---
    auto params = block.parameters();
    int total = 0;
    for (auto& p : params) total += p->numel();

    std::cout << "Architecture:\n";
    std::cout << "  seq_len : " << seq_len << "\n";
    std::cout << "  d_model : " << d_model << "\n";
    std::cout << "  d_k     : " << d_k     << "\n";
    std::cout << "  d_ff    : " << d_ff    << "\n";
    std::cout << "  total params: " << total << "\n\n";

    // --- forward pass ---
    TensorPtr input = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model});
    input->randomize(0.1f);

    TensorPtr output = block.forward(input);

    // --- print input vs output ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::left
              << std::setw(6)  << "Token"
              << std::setw(50) << "Input (first 4 dims)"
              << std::setw(50) << "Output (first 4 dims)" << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (int i = 0; i < seq_len; i++) {
        std::cout << std::setw(6) << i;
        // input
        std::string in_str = "";
        for (int j = 0; j < 4; j++)
            in_str += std::to_string(input->data[i * d_model + j]).substr(0, 8) + " ";
        std::cout << std::setw(50) << in_str;
        // output
        std::string out_str = "";
        for (int j = 0; j < 4; j++)
            out_str += std::to_string(output->data[i * d_model + j]).substr(0, 8) + " ";
        std::cout << std::setw(50) << out_str << "\n";
    }

    // --- verify output shape matches input shape ---
    std::cout << "\nShape check:\n";
    std::cout << "  input  shape: [" << input->shape[0]  << ", " << input->shape[1]  << "]\n";
    std::cout << "  output shape: [" << output->shape[0] << ", " << output->shape[1] << "]\n";
    bool shape_ok = (input->shape == output->shape);
    std::cout << (shape_ok ? "[PASS]" : "[FAIL]") << " output shape matches input shape\n";

    // --- gradient flow check ---
    std::cout << "\nGradient flow check:\n";
    input->requires_grad = true;
    TensorPtr out2 = block.forward(input);
    TensorPtr loss = sum(out2);
    loss->backward();
    bool grad_ok = (input->grad != nullptr);
    std::cout << (grad_ok ? "[PASS]" : "[FAIL]") << " gradients reach input\n";

    // --- max grad magnitude ---
    if (grad_ok) {
        float max_grad = 0.0f;
        for (int i = 0; i < input->grad->numel(); i++)
            max_grad = std::max(max_grad, std::abs(input->grad->data[i]));
        std::cout << "  max |grad| at input: " << max_grad << "\n";
    }

    return 0;
}