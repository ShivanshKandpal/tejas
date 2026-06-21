#pragma once

#include "tensor.h"

TensorPtr cuda_matmul_tiled_wrapper(const TensorPtr& a, const TensorPtr& b);
TensorPtr cuda_add_wrapper(const TensorPtr& a, const TensorPtr& b);
TensorPtr cuda_relu_wrapper(const TensorPtr& a);
TensorPtr cuda_multiply_wrapper(const TensorPtr& a, const TensorPtr& b);
TensorPtr cuda_softmax_wrapper(const TensorPtr& a);
TensorPtr cuda_scale_wrapper(const TensorPtr& a, float scalar);
TensorPtr cuda_transpose_wrapper(const TensorPtr& a);
TensorPtr cuda_scale_wrapper(const TensorPtr& a, float scalar);