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

    class SingleHeadAttention {
        public: 
            Linear q_proj;
            Linear v_proj;
            Linear k_proj;
            Linear o_proj;
            float d_k;
            
            SingleHeadAttention(int d_model, int d_k);

            TensorPtr forward(const TensorPtr& x);
            std::vector<TensorPtr> parameters() const;
            void cpu();
            void cuda();
    };

    class FeedForward {
        public:
            Linear w1;
            Linear w2;

            FeedForward(int d_model, int d_ff);
            
            TensorPtr forward(const TensorPtr& x);

            std::vector<TensorPtr> parameters() const;
            void cpu();
            void cuda();
    };

    class TransformerBlock {
        public: 
            LayerNorm ln1;
            SingleHeadAttention attn;
            LayerNorm ln2;
            FeedForward ffn;

            TransformerBlock(int d_model, int d_k, int d_ff);

            TensorPtr forward(const TensorPtr& x);

            std::vector<TensorPtr> parameters() const;
            void cpu();
            void cuda();
    };
}

