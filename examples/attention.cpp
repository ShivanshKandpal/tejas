#include "tensor.h"
#include <iostream>
#include <cmath>    

int main() {
    int seq_len = 4;
    int d_model = 8;
    int d_k     = 4;
    TensorPtr W_q = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_q->randomize();
    TensorPtr W_k = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_k->randomize();
    TensorPtr W_v = std::make_shared<Tensor>(std::vector<int>{d_model, d_k}); W_v->randomize();

    TensorPtr input = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model}); input->randomize();

    TensorPtr Q = matmul(input, W_q);
    TensorPtr K = matmul(input, W_k);
    TensorPtr V = matmul(input, W_v);

    TensorPtr scores = matmul(Q, transpose(K));
    scores = scale(scores, 1.0f/std::sqrt(d_k));

    TensorPtr attn = softmax(scores);
    
    TensorPtr output = matmul(attn, V);

    output->print();


}