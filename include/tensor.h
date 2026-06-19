#pragma once
#include <vector>
#include <functional>
#include <memory>
#include "device.h"

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
struct Tensor : public std::enable_shared_from_this<Tensor> {
    std::vector<int> shape;
    std::vector<int> strides;

    std::vector<float> data;
    float* gpu_data = nullptr;
    Device device = Device::CPU;

    std::vector<TensorPtr> _prev;
    TensorPtr grad = nullptr;
    bool requires_grad = false;
    std::function<void()>  backward_fn = nullptr;

    Tensor(std::vector<int> shape); 
    Tensor(std::vector<int> shape, std::vector<float> values);

    ~Tensor();

    TensorPtr to(Device target_device);
    TensorPtr cpu();
    TensorPtr cuda();

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
    int numel() const {
        if(shape.empty()) return 0;
        int elements = 1;
        for(int s : shape) elements *= s;
        return elements;
    }
    void randomize(float scale = 0.1f); 
    void zero_grad();
       
};
TensorPtr matmul_raw(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul_tiled_raw(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);
TensorPtr matmul_tiled(const TensorPtr& a, const TensorPtr& b);
TensorPtr add(const TensorPtr& a, const TensorPtr& b);
TensorPtr multiply_raw(const TensorPtr& a, const TensorPtr& b);
TensorPtr multiply(const TensorPtr& a, const TensorPtr& b);
TensorPtr relu(const TensorPtr& a);
TensorPtr transpose(const TensorPtr& a);
TensorPtr transpose_raw(const TensorPtr& a);
TensorPtr sum(const TensorPtr& a);
TensorPtr mse_loss(const TensorPtr& pred, float target_val);
void sgd_step(const TensorPtr& param, float lr);


