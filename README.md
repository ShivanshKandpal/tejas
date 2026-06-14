# tejas

A tensor library and autograd engine built from scratch in C++, with CUDA kernels in progress.

## Why

Most people use PyTorch without understanding what's underneath it. This project is an attempt to build the core of a tensor library including memory layout, matrix multiplication, automatic differentiation from first principles, in order to actually understand how frameworks like PyTorch work at the metal level.

## What's implemented

**Tensor core**
- N-dimensional tensor backed by a flat `std::vector<float>`, with shape and stride-based indexing
- Stride computation supports arbitrary dimensionality

**Matrix multiplication**
- Naive O(n³) implementation
- Tiled implementation (cache-blocked, tile size 32)
- Benchmarked against each other across sizes 64–1024

**Elementwise ops**
- `add`, `multiply`, `relu`, `transpose`, `sum`

**Autograd engine**
- Reverse-mode automatic differentiation over a dynamically built computation graph
- Each tensor stores `shared_ptr` references to its inputs (`_prev`), which keeps the graph alive for backward
- `backward()` performs a topological sort and calls each node's `backward_fn` in reverse order
- Gradients implemented for matmul, add, multiply, relu, transpose, sum

**Training demo**
- A 2-layer MLP trained on XOR using only this library — `matmul → add → relu → matmul → add → MSE loss`, trained with SGD
- Converges from loss ~1.68 to ~0 within 100 epochs

## Benchmarks

Naive vs tiled matmul (square matrices, -O2):

| Size | Naive (ms) | Tiled (ms) | Speedup |
|------|-----------|------------|---------|
| 64   | 0.32      | 0.37       | 0.86x   |
| 128  | 2.50      | 2.54       | 0.98x   |
| 256  | 30.05     | 19.54      | 1.53x   |
| 512  | 245.928   | 199.34     | 1.23x   |
| 1024 | 2644.25   | 2232.56    | 1.18x   |

Note: the initial tiled implementation was *not* faster than naive due to overhead in the generic `at()` indexing function (a `std::vector` allocation on every element access). Switching to a direct-index `at2d()` and using tile size 32 produced the numbers above. This was a useful lesson in measuring rather than assuming where bottlenecks are.

Note : approximate, measured with i7-9750H CPU on WSL2

## Design notes

**Why `shared_ptr` everywhere**

Backward functions need to reference the tensors that produced them, but those tensors may go out of scope before `backward()` is called. Each tensor holds `shared_ptr` references to its inputs (`_prev`), which keeps the entire computation graph alive for as long as the output tensor is alive — the same approach PyTorch uses internally (`shared_ptr<TensorImpl>`).

**Raw vs autograd-tracked ops**

Internal operations (`matmul_raw`, `transpose_raw`, etc.) perform pure computation with no graph tracking. The public versions (`matmul`, `transpose`, etc.) wrap these and additionally set `_prev` and `backward_fn`. This separation prevents backward passes from accidentally building new graph nodes during gradient computation.

## Build & run

```bash
# run tests
g++ tests/test_tensor.cpp src/tensor.cpp -I include -O2 -o test_tensor
./test_tensor

# run benchmark
g++ benchmarks/bench_matmul.cpp src/tensor.cpp -I include -O2 -o bench_matmul
./bench_matmul

# run XOR training demo
g++ examples/xor.cpp src/tensor.cpp -I include -O2 -o xor
./xor
```

## Roadmap

- [x] Tensor struct with strides
- [x] Naive + tiled matmul with benchmarks
- [x] Elementwise ops (add, multiply, relu, transpose, sum)
- [x] Autograd engine (shared_ptr-based computation graph)
- [x] XOR training demo
- [ ] CUDA kernels (vector add, matmul)
- [ ] CUDA tiled matmul with shared memory
- [ ] GPU vs CPU benchmarks
- [ ] Small transformer forward pass