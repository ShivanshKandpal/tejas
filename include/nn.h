#pragma once
#include "tensor.h"
#include <vector>

namespace tejas::nn {
    class Linear {
        public:
            TensorPtr weight;
            TensorPtr bias;

            Linear(int in_features, int out_features);

            void cuda();
            void cpu();
            TensorPtr forward(const TensorPtr& x);
            
            std::vector<TensorPtr> parameters() const;

    };
}