#pragma once
#include<vector>

struct Tensor{
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;
    Tensor(std::vector<int> shape); 
    Tensor(std::vector<int> shape, std::vector<float> values);
    float& at(std::vector<int> indices);
    float at(std::vector<int> indices) const;
    void print();
    inline float& at2d(int i, int j){
        return data[i* strides[0] + j * strides[1]];
    }
    inline float at2d(int i, int j) const{
        return data[i* strides[0] + j * strides[1]];
    } 
    int numel() const { return data.size(); }
};
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_tiled(const Tensor& a, const Tensor& b);
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor relu(const Tensor& a);
