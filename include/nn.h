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
    class LayerNorm {
        public:
            TensorPtr gamma; 
            TensorPtr beta;  
            float eps;

            LayerNorm(int normalized_shape, float eps = 1e-5f);
            
            TensorPtr forward(const TensorPtr& x);
            
            std::vector<TensorPtr> parameters() const;
            
            void cuda();
            void cpu();
        };
}

