#pragma once

#include "tensor.h"

TensorPtr cuda_matmul_tiled_wrapper(const TensorPtr& a, const TensorPtr& b);