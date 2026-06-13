#include "tensor.h"
#include<iostream>
#include<cassert>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
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
TensorPtr matmul(TensorPtr a, TensorPtr b){
    std::vector<int> shapea = a->shape;
    std::vector<int> shapeb = b->shape;
    assert(shapea[1] == shapeb[0]);
    // std::vector<int> shapec = {shapea[0],shapeb[1]};
    TensorPtr c = std::make_shared<Tensor>(std::vector<int> {shapea[0],shapeb[1]});
    c->_prev = {a,b};
    for(int i = 0;i<shapea[0];i++){
        for(int j = 0;j<shapeb[1];j++){
            for(int k = 0;k<shapea[1];k++){
                c->at2d(i,j) += a->at2d(i,k)*b->at2d(k,j);
            }
        }
    }
    if(a->requires_grad || b->requires_grad){
        c->requires_grad = true;
        c->backward_fn = [a, b, c](){
            if(a->requires_grad){
                TensorPtr dA = matmul(c->grad, transpose(b));
                if(a->grad == nullptr) a->grad = std::make_shared<Tensor>(a->shape);
                for(int i = 0;i<a->numel();i++){
                    a->grad->data[i] += dA->data[i];
                }
            }
            if(b->requires_grad){
                TensorPtr dB = matmul(transpose(a), c->grad);
                if(b->grad == nullptr) b->grad = std::make_shared<Tensor>(b->shape);
                for(int i = 0;i<b->numel();i++){
                    b->grad->data[i] += dB->data[i];
                }
            }
        };  
    }
    return c;
}

TensorPtr matmul_tiled(TensorPtr a, TensorPtr b){
    int tile = 16;
    assert(a->shape[1] == b->shape[0]);
    int M = a->shape[0];
    int N = b->shape[1];
    int K = a->shape[1];
    TensorPtr c = std::make_shared<Tensor> (std::vector<int> {M,N});
    for(int tile_i=0;tile_i<M;tile_i+=tile){
        for(int tile_j=0;tile_j<N;tile_j+=tile){
            for(int tile_k=0;tile_k<K;tile_k+=tile){
                int i_end = std::min(tile_i+tile,M);
                int j_end = std::min(tile_j+tile,N);
                int k_end = std::min(tile_k+tile,K);
                for(int i = tile_i ;i<i_end;i++){
                    for(int j = tile_j;j<j_end;j++){
                        for(int k = tile_k;k<k_end;k++){
                            c->at2d(i,j) += a->at2d(i,k) * b->at2d(k,j);
                        }
                    }
                }
            }
        }
    }
    return c;
}

TensorPtr add(TensorPtr a, TensorPtr b){
    assert(a->shape == b->shape);
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i =0;i<a->numel();i++){
        result->data[i] = a->data[i] + b->data[i];
    }
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
                for(int i = 0;i<b->numel();i++){
                    b->grad->data[i] += result->grad->data[i];
                }
            }
        };
    }
    return result;
}
TensorPtr multiply(TensorPtr a, TensorPtr b){
    assert(a->shape == b->shape);
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i =0;i<a->numel();i++){
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}
TensorPtr relu(TensorPtr a){
    
    TensorPtr result = std::make_shared<Tensor>(a->shape);
    for(int i = 0;i<a->numel();i++){
        result->data[i] = std::max(0.0f, a->data[i]);
    }
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

TensorPtr sum(TensorPtr a){
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

TensorPtr transpose(TensorPtr a){
    assert(a->shape.size() == 2);
    TensorPtr result = std::make_shared<Tensor>(std::vector<int>{a->shape[1],a->shape[0]});
    for(int i = 0;i<a->shape[0];i++){
        for(int j = 0;j<a->shape[1];j++){
            result->at2d(j,i) = a->at2d(i,j);
        }
    }
    return result;
}