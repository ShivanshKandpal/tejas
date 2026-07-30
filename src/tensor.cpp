#include "tensor.h"
#ifdef USE_CUDA
    #include <cuda_runtime.h> 
    #include "backend.h"
#endif
#include <iostream>
#include <cassert>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include <random>

Tensor::Tensor(std::vector<int> shape){
    this->shape = shape;
    int total = 1;
    for(int i = 1;i<shape.size();i++){
        total*=shape[i];
    }
    int data_size = total *shape[0];
    this->data.resize(data_size, 0.0f);
    for(int i = 1;i<shape.size();i++){
        this->strides.push_back(total);
        total/=shape[i];
    }
    this->strides.push_back(1);
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> values){
    this->shape = shape;
    this->data = values;
    int total = 1;
    for(int i = 1;i<shape.size();i++) total*=shape[i];
    for(int i = 1;i<shape.size();i++) {
        this->strides.push_back(total); 
        total/=shape[i];
    }
    this->strides.push_back(1);
}

Tensor::~Tensor() {
#ifdef USE_CUDA
    
    if (gpu_data != nullptr) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }
#endif
}

TensorPtr Tensor::to(Device target){
    if (this->device == target){
       return shared_from_this();
    }
    
    TensorPtr result = std::make_shared<Tensor>(this->shape);
    result->device = target;

    int total_bytes = this->numel() * sizeof(float);

    if(target == Device::CUDA){
    #ifdef USE_CUDA        
        //cpu to gpu
        cudaMalloc(&result->gpu_data, total_bytes);
        cudaMemcpy(result->gpu_data, this->data.data(), total_bytes, cudaMemcpyHostToDevice);
        result->data.clear();
    #endif
    }
    else if (target == Device::CPU){
        result->data.resize(this->numel());
        #ifdef USE_CUDA
        if (this->device == Device::CUDA) {
            cudaMemcpy(result->data.data(), this->gpu_data, total_bytes, cudaMemcpyDeviceToHost);
        }
        #endif
    }

    result->_prev = {shared_from_this()};
    if(this->requires_grad) {
        result->requires_grad = true;
        result->backward_fn = [this_node = shared_from_this(), result]() {
            if (this_node->grad == nullptr) {
                this_node->grad = std::make_shared<Tensor>(this_node->shape);
            }
        };
    }
    return result;
}

TensorPtr Tensor::cpu() {
    return this->to(Device::CPU);

}

TensorPtr Tensor::cuda() {
    return this->to(Device::CUDA);
}

float& Tensor::at(std::vector<int> indices){
    int index = 0;
    for(int i = 0;i<indices.size();i++){
        index += indices[i]*this->strides[i];
    }
    return data[index];
}
float Tensor::at(std::vector<int> indices) const{
    int index = 0;
    for(int i = 0;i<indices.size();i++){
        index += indices[i]*this->strides[i];
    }
    return data[index];
}
void Tensor::print(){

    if(this->device == Device::CUDA) {
        std::cout << "[GPU Tensor] -> ";
        auto cpu_copy = this->to(Device::CPU);
        cpu_copy->print();
        return;
    }
    std::cout<<"shape : [";
    for(int i = 0;i<this->shape.size();i++){
        if(i == this->shape.size() - 1){
            std::cout<<this->shape[i];
            continue;
        }
        std::cout<<this->shape[i]<<", ";
    }
    std::cout<<"]\n";
    std::cout<<"values: [";
    for(int i = 0;i<this->data.size();i++){
        if(i == this->data.size() - 1){
            std::cout<<this->data[i];
            continue;
        }
        std::cout<<this->data[i]<<", ";
    }
    std::cout<<"]\n";
}
TensorPtr matmul_raw(const TensorPtr& a, const TensorPtr& b){
    std::vector<int> shapea = a->shape;
    std::vector<int> shapeb = b->shape;
    assert(shapea[1] == shapeb[0]);

    if (a->device != b->device) {
        throw std::runtime_error("Runtime Error: Tensors must be on the same device!");
    }

    if(a->device == Device::CUDA) {
#ifdef USE_CUDA
        return cuda_matmul_tiled_wrapper(a, b);
#else 
        throw std::runtime_error("Tejas was compiled without CUDA support. ");
#endif
    }
    TensorPtr result = std::make_shared<Tensor>(std::vector<int> {shapea[0],shapeb[1]});
    for(int i = 0; i < shapea[0]; i++){
        for(int k = 0; k < shapea[1]; k++){
            float a_ik = a->at2d(i, k); 
            for(int j = 0; j < shapeb[1]; j++){
                result->at2d(i, j) += a_ik * b->at2d(k, j); 
            }
        }
    }
    return result;
}
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b){
    std::vector<int> shapea = a->shape;
    std::vector<int> shapeb = b->shape;
    assert(shapea[1] == shapeb[0]);
    TensorPtr result = matmul_raw(a,b);
    result->_prev = {a,b};
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, b, result](){
            if(a->requires_grad){
                TensorPtr dA = matmul_raw(result->grad, transpose_raw(b));
                if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
                for(int i = 0;i<a->numel();i++){
                    a->grad->data[i] += dA->data[i];
                }
            }
            if(b->requires_grad){
                TensorPtr dB = matmul_raw(transpose_raw(a), result->grad);
                if(b->grad == nullptr) b->grad = std::make_shared<Tensor>(b->shape);
                for(int i = 0;i<b->numel();i++){
                    b->grad->data[i] += dB->data[i];
                }
            }
        };  
    }
    return result;
}

TensorPtr matmul_tiled_raw(const TensorPtr& a, const TensorPtr& b){
    int tile = 32;
    assert(a->shape[1] == b->shape[0]);
    int M = a->shape[0];
    int N = b->shape[1];
    int K = a->shape[1];
    TensorPtr result = std::make_shared<Tensor> (std::vector<int> {M,N});
    for(int tile_i=0; tile_i<M; tile_i+=tile){
        for(int tile_j=0; tile_j<N; tile_j+=tile){
            for(int tile_k=0; tile_k<K; tile_k+=tile){
                int i_end = std::min(tile_i+tile, M);
                int j_end = std::min(tile_j+tile, N);
                int k_end = std::min(tile_k+tile, K);
                for(int i = tile_i; i < i_end; i++){
                    for(int k = tile_k; k < k_end; k++){
                        float a_ik = a->at2d(i, k); 
                        for(int j = tile_j; j < j_end; j++){
                            result->at2d(i, j) += a_ik * b->at2d(k, j);
                        }
                    }
                }
            }
        }
    }
    return result;
}


TensorPtr matmul_tiled(const TensorPtr& a, const TensorPtr& b){
    assert(a->shape[1] == b->shape[0]);
    TensorPtr result = matmul_tiled_raw(a,b);
    result->_prev = {a,b};
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, b, result](){
            if(a->requires_grad){
                TensorPtr dA = matmul_raw(result->grad, transpose_raw(b));
                if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
                for(int i = 0;i<a->numel();i++){
                    a->grad->data[i] += dA->data[i];
                }
            }
            if(b->requires_grad){
                TensorPtr dB = matmul_raw(transpose_raw(a), result->grad);
                if(b->grad == nullptr) b->grad = std::make_shared<Tensor>(b->shape);
                for(int i = 0;i<b->numel();i++){
                    b->grad->data[i] += dB->data[i];
                }
            }
        };  
    }
    return result;
}

TensorPtr add_raw(const TensorPtr& a, const TensorPtr& b) {
    // assert(a->shape == b->shape);
    if(a->device != b->device) throw std::runtime_error("Device mismatch in add. ");

    if(a->device == Device::CUDA) {
#ifdef USE_CUDA
        return cuda_add_wrapper(a, b);
#else 
        throw std::runtime_error("Tejas was compiled without CUDA support.");
#endif  
    }
    if(a->shape == b->shape) {
        TensorPtr result = std::make_shared<Tensor>(a->shape);
        for(int i =0;i<a->numel();i++){
            result->data[i] = a->data[i] + b->data[i];
        }
        return result;
    }

    else if (a->shape.size() == 2 && b->shape.size() <= 2 && b->shape.back() == a->shape[1]) {
        TensorPtr result = std::make_shared<Tensor>(a->shape);
        int N = a->shape[1];
        for(int i = 0;i < a->numel(); i++) {
            result->data[i] = a->data[i] + b->data[i % N];
        }
        return result;
    }
    else {
        throw std::runtime_error("Shapes are not compatible for addition or broadcasting. ");   
    }
}

TensorPtr add(const TensorPtr& a, const TensorPtr& b){
    TensorPtr result = add_raw(a, b);
    result->_prev = {a,b};
    if(a->requires_grad || b-> requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, b, result](){
            if(a->requires_grad){
                if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
                for(int i = 0;i<a->numel();i++){
                    a->grad->data[i] += result->grad->data[i];
                }
            }
            if(b->requires_grad){
                if(b->grad == nullptr) b->grad = std::make_shared<Tensor>(b->shape);
                if(a->shape != b->shape) {
                    int N = a->shape[1];
                    for(int i = 0; i < a->numel(); i++) {
                        b->grad->data[i % N] += result->grad->data[i];
                    }
                }
                else {
                    for(int i = 0;i<b->numel();i++){
                        b->grad->data[i] += result->grad->data[i];
                    }
                }
            }
        };
    }
    return result;
}
TensorPtr multiply_raw(const TensorPtr& a, const TensorPtr& b){
    assert(a->shape == b->shape);
    if(a->device != b->device) throw std::runtime_error("Device mismatch in multiply. ");
    if(a->device == Device::CUDA) {
#ifdef USE_CUDA
        return cuda_multiply_wrapper(a, b);
#else 
        throw std::runtime_error("Tejas was compiled without CUDA support.");
#endif
    }
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i =0;i<a->numel();i++){
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b){
    TensorPtr result = multiply_raw(a,b);
    result->_prev = {a,b};
    if(a->requires_grad || b->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, b, result](){
            if(a->requires_grad){
                if(a->grad ==  nullptr) a->grad = std::make_shared<Tensor>(a->shape);
                TensorPtr dA = multiply_raw(result->grad, b);
                for(int i = 0;i < a->numel();i++){
                    a->grad->data[i] += dA->data[i];
                }
            }
            if (b->requires_grad) {
                if (b->grad == nullptr) b->grad = std::make_shared<Tensor>(b->shape);
                TensorPtr dB = multiply_raw(result->grad, a);
                for(int i = 0; i < b->numel(); i++) {
                    b->grad->data[i] += dB->data[i];
                }
            }
        };
    }
    return result;
}

TensorPtr relu_raw(const TensorPtr& a) {
    if(a->device == Device::CUDA) {
#ifdef USE_CUDA
        return cuda_relu_wrapper(a);
#else 
        throw std::runtime_error("Tejas was compiled without CUDA support.");
#endif
    }
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i = 0;i<a->numel();i++){
        result->data[i] = std::max(0.0f, a->data[i]);
    }
    return result;
}

TensorPtr relu(const TensorPtr& a){
    TensorPtr result = relu_raw(a);
    result->_prev = {a};
    if(a->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, result](){
            if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
            for(int i = 0;i<a->numel();i++){
                a->grad->data[i] += result->grad->data[i] * (a->data[i]>0?1.0f:0.0f);
            }
        };
    }
    return result;
}

TensorPtr sum(const TensorPtr& a){
    TensorPtr result = std::make_shared<Tensor>(std::vector<int>{1});
    for(int i = 0;i<a->numel();i++){
        result->data[0] += a->data[i];
    }
    result->_prev = {a};
    if(a->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, result](){
            if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
            for(int i = 0;i<a->numel();i++){
                a->grad->data[i] += result->grad->data[0] * 1.0f;
            }
        };
    }
    return result;
}

void Tensor::backward(){
    if(this->device == Device::CUDA) {
        throw std::runtime_error(
            "Runtime Error: backward() is not yet supported directly on GPU tensors. "  
            "Call .cpu() on your loss tensor before calling backward(). Full GPU autograd is planned. "
        );
    }

    std::vector<TensorPtr> topo;
    std::set<Tensor*> visited;
    std::function<void(TensorPtr)> build = [&](TensorPtr node){
        if(visited.count(node.get())) return;
        visited.insert(node.get());
        for(auto& child : node->_prev){
            build(child);
        }
        topo.push_back(node);
    };
    build(shared_from_this());
    
    grad = std::make_shared<Tensor> (shape);
    for(int i = 0;i < numel();i++) grad->data[i] = 1.0f;
    for(auto it = topo.rbegin();it!=topo.rend();it++){
        if((*it)->backward_fn) (*it)->backward_fn();
    }
}

TensorPtr transpose_raw(const TensorPtr& a){
    assert(a->shape.size() == 2);
    if(a->device == Device::CUDA) {
        #ifdef USE_CUDA
            return cuda_transpose_wrapper(a);
        #else
            throw std::runtime_error("Tejas was not compiled with CUDA support. ");
        #endif
    }
    TensorPtr result = std::make_shared<Tensor>(std::vector<int>{a->shape[1],a->shape[0]});
    for(int i = 0;i<a->shape[0];i++){
        for(int j = 0;j<a->shape[1];j++){
            result->at2d(j,i) = a->at2d(i,j);
        }
    }
    return result;
}

TensorPtr transpose(const TensorPtr& a){
    TensorPtr result = transpose_raw(a);
    result->_prev = {a};
    if(a->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, result](){
            if (a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
            TensorPtr dL_dA = transpose_raw(result->grad);
            for(int i = 0;i < a->numel(); i++){
                a->grad->data[i] += dL_dA->data[i];
            }
        };  
    }
    return result;
}
void Tensor::randomize(float scale){
    static std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for(int i = 0;i < numel();i++){
        data[i] = dist(gen) * scale;
    }   
}
void Tensor::zero_grad(){
    grad = nullptr;
}

TensorPtr softmax_raw(const TensorPtr& a) {
    assert(a->shape.size() <= 2);
    if(a->device == Device::CUDA) {
        #ifdef USE_CUDA
            return cuda_softmax_wrapper(a);
        #else
            throw std::runtime_error("Tejas was compiled without cuda support.");
        #endif
    }
    TensorPtr result = std::make_shared<Tensor>(a->shape);

    //rn a can only be 1D or 2D 
    int rows = (a->shape.size() > 1) ? a->shape[0] : 1;
    int cols = (a->shape.size() > 1) ? a->shape[1] : a->shape[0];

    for(int i = 0; i < rows; i++) {
        int row_offset = i * cols;

        float max_val = -std::numeric_limits<float>::infinity();
        for(int j = 0; j < cols; j++) {
            max_val = std::max(max_val, a->data[row_offset + j]);
        }

        float sum = 0.0f;
        for(int j = 0; j < cols; j++) {
            float e = std::exp(a->data[row_offset + j] - max_val);
            result->data[row_offset + j] = e;
            sum += e;
        }

        for(int j = 0; j < cols; j++) {
            result->data[row_offset + j] /= sum;
        }
    }
    return result;
}

TensorPtr softmax(const TensorPtr& a) {
    TensorPtr result = softmax_raw(a);
    
    result->_prev = {a};
    if(a->requires_grad) {
        result->requires_grad = true;
        result->backward_fn = [a, result](){
            if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
            
            int rows = (a->shape.size() > 1) ? a->shape[0] : 1;
            int cols = (a->shape.size() > 1) ? a->shape[1] : a->shape[0];

            for(int i = 0; i < rows; i++) {
                int row_offset = i * cols;

                float dot = 0.0f;
                
                for(int j = 0;j < cols; j++) {
                    float rd = result->data[row_offset + j];
                    float rgd = result->grad->data[row_offset + j];

                    dot += rd * rgd;
                }

                for(int j = 0; j < cols; j++) {
                    float rd = result->data[row_offset + j];
                    float rgd = result->grad->data[row_offset + j];

                    a->grad->data[row_offset + j] += rd * (rgd - dot);
                }

            }
        };
    }
    return result;

}

TensorPtr mse_loss(const TensorPtr& pred, float target_value){
    TensorPtr loss = std::make_shared<Tensor>(std::vector<int> {1});
    float diff = pred->data[0] - target_value;
    loss->data[0] = diff * diff;
    loss->_prev = {pred};
    if(pred->requires_grad){
        loss->requires_grad = true;
        loss->backward_fn = [pred, diff](){
            if(pred->grad == nullptr) pred->grad = std::make_shared<Tensor>(pred->shape);
            pred->grad->data[0] += 2.0f * diff;
        };
    }
    return loss;
}
void sgd_step(const TensorPtr& param, float lr){
    for(int i = 0; i < param->numel(); i++){
        param->data[i] -= lr * param->grad->data[i];
    }
}

TensorPtr scale_raw(const TensorPtr& a, float scalar) {
    if(a->device == Device::CUDA) {
        #ifdef USE_CUDA
            return cuda_scale_wrapper(a, scalar);
        #else 
            throw std::runtime_error("Tejas was not compiled with CUDA support. ");
        #endif
    }

    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i = 0; i < a->numel(); i++){
        result->data[i] = a->data[i] * scalar;
    }
    
    return result;
}

TensorPtr scale(const TensorPtr& a, float scalar) {
    
    TensorPtr result = scale_raw(a, scalar);
    result->_prev = {a};

    if(a->requires_grad){
        result->requires_grad = true;
        result->backward_fn = [a, result, scalar](){
            if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
            for(int i = 0; i < a->numel(); i++) {
                a->grad->data[i] += result->grad->data[i] * scalar;
            }
        };
    }
    return result;
};

TensorPtr layernorm(const TensorPtr& x, const TensorPtr& gamma, const TensorPtr& beta, float eps) {
    int M = x->shape[0];
    int N = x->shape[1];
    TensorPtr out = std::make_shared<Tensor>(x->shape);

    std::shared_ptr<std::vector<float>> x_hat = std::make_shared<std::vector<float>>(x->numel());
    std::shared_ptr<std::vector<float>> inv_std = std::make_shared<std::vector<float>>(M);

    for(int i = 0; i < M; i++) {
        float mean = 0.0f;
        for(int j = 0; j < N; j++) mean += x->data[i * N + j];
        mean /= N;

        float var = 0.0f;
        for(int j = 0; j < N; j++) {
            float diff = x->data[i * N + j] - mean;
            var += diff * diff;
        }
        var /= N;

        float inv_stddev = 1.0f / std::sqrt(var + eps);
        (*inv_std)[i] = inv_stddev;
        
        for(int j = 0; j < N; j++) {
            int idx = i * N + j;
            (*x_hat)[idx] = (x->data[idx] - mean) * inv_stddev; 
            out->data[idx] = (*x_hat)[idx] * gamma->data[j] + beta->data[j];
        }
    }

    out->_prev = {x, gamma, beta};

    if (x->requires_grad || gamma->requires_grad || beta->requires_grad) {
        out->requires_grad = true;
        out->backward_fn = [x, gamma, beta, out, x_hat, inv_std, M, N]() {
            if(gamma->requires_grad && gamma->grad == nullptr) gamma->grad = std::make_shared<Tensor>(gamma->shape);
            if(beta->requires_grad && beta->grad == nullptr) beta->grad = std::make_shared<Tensor>(beta->shape);
            if(x->requires_grad && x->grad == nullptr) x->grad = std::make_shared<Tensor>(x->shape);

            for(int i = 0; i < M; i++) {
                float sum_grad_xhat = 0.0f;
                float sum_grad_xhat_xhat = 0.0f;

                for(int j = 0; j < N; j++) {
                    int idx = i * N + j;
                    float grad_out = out->grad->data[idx];
                    float x_hat_val = (*x_hat)[idx];

                    if(gamma->requires_grad) {
                        gamma->grad->data[j] += grad_out * x_hat_val;
                    }

                    if(beta->requires_grad) {
                        beta->grad->data[j] += grad_out;
                    }

                    float grad_xhat = grad_out * gamma->data[j];

                    sum_grad_xhat += grad_xhat;
                    sum_grad_xhat_xhat += grad_xhat * x_hat_val;
                }

                if(x->requires_grad) {
                    float inv = (*inv_std)[i];

                    for(int j = 0; j < N; j++) {
                        int idx = i * N + j;
                        float grad_out = out->grad->data[idx];
                        float grad_xhat = grad_out * gamma->data[j];
                        float x_hat_val = (*x_hat)[idx];
                        
                        x->grad->data[idx] += (inv / N) * (N * grad_xhat - sum_grad_xhat - x_hat_val * sum_grad_xhat_xhat);
                    }   
                }

            }
        };
    }
    return out;
}

TensorPtr gelu_raw(const TensorPtr& a) {

    TensorPtr result = std::make_shared<Tensor>(a->shape);

    for(int i = 0; i < a->numel(); i++) {
        float val = a->data[i];
        float sig = 1.0f / (1.0f + std::exp(-1.702f * val));
        result->data[i] = val * sig;
    }
    return result;
}

TensorPtr gelu(const TensorPtr& a) {
    TensorPtr result = gelu_raw(a);

    result->_prev = {a};
    if(a->requires_grad) {
        result->requires_grad = true;
        result->backward_fn = [a, result] {
            if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);

            for(int i = 0; i < a->numel(); i++) {
                float x = a->data[i];
                float sig = (1.0f / (1.0f + std::exp(-1.702 * x)));
                float local_grad = sig * (1 + x * 1.702 * (1-sig));
                a->grad->data[i] += result->grad->data[i] * local_grad;
            }
        };
    }
    return result;
}

TensorPtr cross_entropy_loss(const TensorPtr& logits, const std::vector<int>& target_indices) {
    int batch_size = logits->shape[0];
    int vocab_size = logits->shape[1];

    TensorPtr loss = std::make_shared<Tensor>(std::vector<int> {1});

    std::shared_ptr<std::vector<float>> probs = std::make_shared<std::vector<float>> (logits->numel());

    float total_loss = 0.0f;

    for(int i = 0; i < batch_size; i++) {
        float max_val = -1e9f;
        for(int j = 0;j < vocab_size; j++) {
            max_val = std::max(max_val, logits->data[i * vocab_size + j]);
        }

        float sum_exp = 0.0f;
        for(int j = 0; j < vocab_size; j++) {
            float e = std::exp(logits->data[i * vocab_size + j] - max_val);
            (*probs)[i * vocab_size + j] = e;
            sum_exp += e;
        }

        int target_idx = target_indices[i];
        if (target_idx < 0 || target_idx >= vocab_size)
            throw std::runtime_error("Target index out of range.");
        
        float target_prob = 0.0f;
        for(int j = 0; j < vocab_size; j++) {
            (*probs)[i * vocab_size + j] /= sum_exp;
            if(j == target_idx) {
                target_prob = (*probs)[i * vocab_size + j];
            }
        }

        total_loss += -std::log(target_prob + 1e-7);
    }

    loss->data[0] = total_loss / batch_size;
    loss->_prev = {logits};

    if(logits->requires_grad) {
        loss->requires_grad = true;
        loss->backward_fn = [logits, probs, target_indices, batch_size, vocab_size, loss]() {
            if(logits->grad == nullptr) logits->grad = std::make_shared<Tensor>(logits->shape);

            float upstream_grad = loss->grad->data[0];

            for(int i = 0; i < batch_size; i++) {
                int target_idx = target_indices[i];

                for(int j = 0; j < vocab_size; j++) {
                    int flat_idx = i * vocab_size + j;
                    float local_grad = (*probs)[flat_idx];
                    
                    if(j == target_idx) {
                        local_grad -= 1.0f;
                    }

                    logits->grad->data[flat_idx] += (local_grad / batch_size) * upstream_grad;
                }

            }
        };
    }
    return loss;
}