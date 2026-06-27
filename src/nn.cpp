#include "nn.h"

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
