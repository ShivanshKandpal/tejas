#pragma once
#include<vector>

struct Tensor{
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;
    Tensor(std::vector<int> shape); 
    float& at(std::vector<int> indices);
    void print();
};