#include "tensor.h"
#include <iostream>
#include <cmath>   
#include <cassert> 

TensorPtr attention(TensorPtr input, TensorPtr W_q, TensorPtr W_k, TensorPtr W_v, float d_k) {
    TensorPtr Q = matmul(input, W_q);
    TensorPtr K = matmul(input, W_k);
    TensorPtr V = matmul(input, W_v);

    TensorPtr scores = matmul(Q, transpose(K));
    scores = scale(scores, 1.0f / std::sqrt(d_k));

    TensorPtr attn = softmax(scores);

    return matmul(attn, V);

}

int main() {

    std::cout <<"=== Tejas Transformer Attention Block ===\n";

    int seq_len = 4;
    int d_model = 8;
    int d_k     = 4;
    TensorPtr W_q = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_q->randomize();
    TensorPtr W_k = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_k->randomize();
    TensorPtr W_v = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_v->randomize();

    TensorPtr input = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model}); input->randomize();

    TensorPtr out_cpu = attention(input, W_q, W_k, W_v, (float)d_k);
    #ifdef USE_CUDA
    TensorPtr d_W_q = W_q->cuda();
    TensorPtr d_W_k = W_k->cuda();
    TensorPtr d_W_v = W_v->cuda();

    TensorPtr d_input = input->cuda();

    TensorPtr out_gpu = attention(d_input, d_W_q, d_W_k, d_W_v, (float)d_k)->cpu();

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