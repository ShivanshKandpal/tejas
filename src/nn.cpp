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

}