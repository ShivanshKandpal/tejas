#pragma once
#include<vector>
#include<functional>
#include <memory>
struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
struct Tensor : public std::enable_shared_from_this<Tensor> {
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;
    std::vector<TensorPtr> _prev;
    TensorPtr grad = nullptr;
    bool requires_grad = false;
    std::function<void()>  backward_fn = nullptr;
    Tensor(std::vector<int> shape); 
    Tensor(std::vector<int> shape, std::vector<float> values);
    float& at(std::vector<int> indices);
    float at(std::vector<int> indices) const;
    void print();
    void backward(); 
    inline float& at2d(int i, int j){
        return data[i* strides[0] + j * strides[1]];
    }
    inline float at2d(int i, int j) const{
        return data[i* strides[0] + j * strides[1]];
    } 
    int numel() const { return data.size(); }
       
};
TensorPtr matmul(TensorPtr a, TensorPtr b);
TensorPtr matmul_tiled(TensorPtr a, TensorPtr b);
TensorPtr add(TensorPtr a, const TensorPtr b);
TensorPtr multiply(TensorPtr a, TensorPtr b);
TensorPtr relu(TensorPtr a);
TensorPtr transpose(TensorPtr a);
TensorPtr sum(TensorPtr a);


