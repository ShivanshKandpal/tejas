#include "tensor.h"
#include "nn.h"
#include <iostream>
#include <cmath>   
#include <cassert> 

TensorPtr attention(TensorPtr input, tejas::nn::Linear& Wq, tejas::nn::Linear& Wk, tejas::nn::Linear& Wv, float d_k) {
    TensorPtr Q = Wq.forward(input);
    TensorPtr K = Wk.forward(input);
    TensorPtr V = Wv.forward(input);

    TensorPtr scores = matmul(Q, transpose(K));
    scores = scale(scores, 1.0f / std::sqrt(d_k));

    TensorPtr attn = softmax(scores);

    return matmul(attn, V);

}

int main() {

    std::cout <<"=== Tejas Transformer Attention Block (with bias) ===\n";

    int seq_len = 4;
    int d_model = 8;
    int d_k     = 4;
    tejas::nn::Linear q_proj(d_model, d_k);
    tejas::nn::Linear k_proj(d_model, d_k);
    tejas::nn::Linear v_proj(d_model, d_k);

    TensorPtr input = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model}); input->randomize();

    TensorPtr out_cpu = attention(input, q_proj, k_proj, v_proj, (float)d_k);
    #ifdef USE_CUDA
    q_proj.cuda();
    k_proj.cuda();
    v_proj.cuda();

    TensorPtr d_input = input->cuda();

    TensorPtr out_gpu = attention(d_input, q_proj, k_proj, v_proj, (float)d_k)->cpu();

    out_cpu->print();
    out_gpu->print();

    bool match = true;
    for(int i = 0; i < out_cpu->numel(); i++) {
        float rel_err = std::abs(out_cpu->data[i] - out_gpu->data[i]) / 
                        (std::abs(out_cpu->data[i]) + 1e-6f);
        
        if(rel_err > 1e-3f) { 
            std::cerr << "[FAIL] Attention mismatch at index " << i 
                      << ": CPU " << out_cpu->data[i] 
                      << " != GPU " << out_gpu->data[i] << "\n";
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "[PASS] Attention Block Hardware Parity Verified!\n";
    } else {
        std::cout << "[FAIL] Attention Block Hardware Parity\n";
        assert(false);
    }
    #else
    std::cout << "[PASS] Attention Block Forward Pass (CPU Only - CI Mode)\n";
    #endif
    return 0;


}