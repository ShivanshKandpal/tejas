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
};
Tensor matmul(const Tensor& a, const Tensor& b);