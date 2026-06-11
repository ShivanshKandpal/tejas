#include "tensor.h"
#include<iostream>
#include<cassert>
#include <algorithm>
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
Tensor matmul(const Tensor&a, const Tensor& b){
    std::vector<int> shapea = a.shape;
    std::vector<int> shapeb = b.shape;
    assert(shapea[1] == shapeb[0]);
    std::vector<int> shapec = {shapea[0],shapeb[1]};
    Tensor c(shapec);

    for(int i = 0;i<shapec[0];i++){
        for(int j = 0;j<shapec[1];j++){
            for(int k = 0;k<shapea[1];k++){
                c.at2d(i,j) += a.at2d(i,k)*b.at2d(k,j);
            }
        }
    }
    return c;
}

Tensor matmul_tiled(const Tensor& a, const Tensor& b){
    int tile = 16;
    assert(a.shape[1] == b.shape[0]);
    int M = a.shape[0];
    int N = b.shape[1];
    int K = a.shape[1];
    Tensor c({M,N});
    for(int tile_i=0;tile_i<M;tile_i+=tile){
        for(int tile_j=0;tile_j<N;tile_j+=tile){
            for(int tile_k=0;tile_k<K;tile_k+=tile){
                int i_end = std::min(tile_i+tile,M);
                int j_end = std::min(tile_j+tile,N);
                int k_end = std::min(tile_k+tile,K);
                for(int i = tile_i ;i<i_end;i++){
                    for(int j = tile_j;j<j_end;j++){
                        for(int k = tile_k;k<k_end;k++){
                            c.at2d(i,j) += a.at2d(i,k) * b.at2d(k,j);
                        }
                    }
                }
            }
        }
    }
    return c;
}

Tensor add(const Tensor& a, const Tensor& b){
    assert(a.shape == b.shape);
    Tensor result(a.shape);
    for(int i =0;i<a.numel();i++){
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}
Tensor multiply(const Tensor& a, const Tensor& b){
    assert(a.shape == b.shape);
    Tensor result(a.shape);
    for(int i =0;i<a.numel();i++){
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}
Tensor relu(const Tensor& a){
    
    Tensor result(a.shape);
    for(int i = 0;i<a.numel();i++){
        result.data[i] = std::max(0.0f, a.data[i]);
    }
    return result;
}