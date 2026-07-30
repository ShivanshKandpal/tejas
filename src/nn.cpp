#include "nn.h"
#include <cmath>

namespace tejas::nn {

    Linear::Linear(int in_features, int out_features) {
        weight = std::make_shared<Tensor>(
            std::vector<int>{in_features, out_features}
        );
        weight->randomize(0.1f);
        weight->requires_grad = true;

        bias = std::make_shared<Tensor>(
            std::vector<int>{1, out_features}
        );
        bias->requires_grad = true;
        
    }

    TensorPtr Linear::forward(const TensorPtr& x) {
        return add(matmul(x, weight), bias);
    }

    std::vector<TensorPtr> Linear::parameters() const {
        return {weight, bias};
    }

    void Linear::cuda() {
        weight = weight->cuda();
        bias = bias->cuda();
    }

    void Linear::cpu() {
        weight = weight->cpu();
        bias = bias->cpu();
    }

    
    LayerNorm::LayerNorm(int normalized_shape, float eps) : eps(eps) {
        gamma = std::make_shared<Tensor>(std::vector<int>{1, normalized_shape});
        for(int i = 0; i < gamma->numel(); i++) gamma->data[i] = 1.0f;
        gamma->requires_grad = true;
        beta = std::make_shared<Tensor>(std::vector<int>{1, normalized_shape});
        beta->requires_grad = true;
    }

    TensorPtr LayerNorm::forward(const TensorPtr& x) {
        return layernorm(x, gamma, beta, eps);
    }

    std::vector<TensorPtr> LayerNorm::parameters() const {
        return {gamma, beta};
    }

    void LayerNorm::cuda() {
        gamma = gamma->cuda();
        beta = beta->cuda();
    }

    void LayerNorm::cpu() {
        gamma = gamma->cpu();
        beta = beta->cpu();
    }

}

namespace tejas::nn {
    SingleHeadAttention::SingleHeadAttention(int d_model, int d_k) : q_proj(d_model, d_k), v_proj(d_model, d_k), k_proj(d_model, d_k), o_proj(d_k, d_model),d_k(d_k) {
    };

    TensorPtr SingleHeadAttention::forward(const TensorPtr& x) {
            TensorPtr Q = q_proj.forward(x);
            TensorPtr K = k_proj.forward(x);
            TensorPtr V = v_proj.forward(x);

            TensorPtr scores = matmul(Q, transpose(K));
            scores = scale(scores, 1.0f / std::sqrt(d_k));

            // --- CAUSAL MASK (No looking into the future) ---
            int seq_len = scores->shape[0]; 
            for (int i = 0; i < seq_len; i++) {
                for (int j = i + 1; j < seq_len; j++) {
                    scores->data[i * seq_len + j] = -1e9f; 
                }
            }

            TensorPtr attn = softmax(scores);

            return o_proj.forward(matmul(attn, V));
    }

    std::vector<TensorPtr> SingleHeadAttention::parameters() const {
        auto pq = q_proj.parameters();
        auto pk = k_proj.parameters();
        auto pv = v_proj.parameters();
        auto po = o_proj.parameters();

        pq.insert(pq.end(), pk.begin(), pk.end());
        pq.insert(pq.end(), pv.begin(), pv.end());
        pq.insert(pq.end(), po.begin(), po.end());
        return pq;
    }

    void SingleHeadAttention::cpu() {
        q_proj.cpu();
        k_proj.cpu();
        v_proj.cpu();
        o_proj.cpu();
    }

    void SingleHeadAttention::cuda() {
        q_proj.cuda();
        k_proj.cuda();
        v_proj.cuda();
        o_proj.cuda();
    }

    FeedForward::FeedForward(int d_model, int d_ff) : w1(d_model, d_ff), w2(d_ff, d_model) {
    }
    TensorPtr FeedForward::forward(const TensorPtr& x) {
        return w2.forward(gelu(w1.forward(x)));
    }

    std::vector<TensorPtr> FeedForward::parameters() const {
        auto p1 = w1.parameters();
        auto p2 = w2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    void FeedForward::cpu() {
        w1.cpu();
        w2.cpu();
    }

    void FeedForward::cuda() {
        w1.cuda();
        w2.cuda();
    }

    TransformerBlock::TransformerBlock(int d_model, int d_k, int d_ff) : ln1(d_model), attn(d_model, d_k), ln2(d_model), ffn(d_model, d_ff) {}

    TensorPtr TransformerBlock::forward(const TensorPtr& x) {
        TensorPtr norm1 = ln1.forward(x);
        TensorPtr attn_out = attn.forward(norm1);
        TensorPtr out1 = add(x, attn_out);

        TensorPtr norm2 = ln2.forward(out1);
        TensorPtr ffn_out = ffn.forward(norm2);
        TensorPtr out2 = add(out1, ffn_out);

        return out2;
    }

    std::vector<TensorPtr> TransformerBlock::parameters() const {
        auto p_ln1 = ln1.parameters();
        auto p_attn = attn.parameters();
        auto p_ln2 = ln2.parameters();
        auto p_ffn = ffn.parameters();
        
        p_ln1.insert(p_ln1.end(), p_attn.begin(), p_attn.end());
        p_ln1.insert(p_ln1.end(), p_ln2.begin(), p_ln2.end());
        p_ln1.insert(p_ln1.end(), p_ffn.begin(), p_ffn.end());

        return p_ln1;
    }

    void TransformerBlock::cuda() { ln1.cuda(); attn.cuda(); ln2.cuda(); ffn.cuda(); }
    void TransformerBlock::cpu()  { ln1.cpu();  attn.cpu();  ln2.cpu();  ffn.cpu(); }

}

namespace tejas::nn {
    Embedding::Embedding(int vocab_size, int d_model) {
        weight = std::make_shared<Tensor>(std::vector<int>{vocab_size, d_model});
        weight->randomize();
        weight->requires_grad = true;
    }

    TensorPtr Embedding::forward(const std::vector<int>& indices) {
        int seq_len = indices.size();
        int d_model = weight->shape[1];

        TensorPtr out = std::make_shared<Tensor>(std::vector<int>{seq_len, d_model});

        for(int i = 0;i < seq_len; i++) {
            int idx = indices[i];
            for(int j = 0; j < d_model; j++) {
                out->data[i * d_model + j] = weight->data[idx * d_model + j];
            }
        }

        out->_prev = {weight};

        if(weight->requires_grad) {
            out->requires_grad = true;
            auto w = weight;
            out->backward_fn = [w , indices, out, seq_len, d_model]() {
                if(w->grad == nullptr) {
                    w->grad = std::make_shared<Tensor>(w->shape);    
                }

                for(int i = 0;i < seq_len; i++) {
                    int idx = indices[i];
                    for(int j = 0; j < d_model; j++) {
                        w->grad->data[idx * d_model + j] += out->grad->data[i * d_model + j];
                    }
                }
            };
        }

        return out;
    }
    std::vector<TensorPtr> Embedding::parameters() const {
        return {weight};
    }

    void Embedding::cuda() {
        weight = weight->cuda();
    };

    void Embedding::cpu() {
        weight = weight->cpu(); 
    };
    
}

