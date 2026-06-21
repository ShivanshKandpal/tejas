# tejas

[![CI](https://github.com/ShivanshKandpal/tejas/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivanshKandpal/tejas/actions/workflows/ci.yml)

A tensor library and autograd engine built from scratch in C++, with custom CUDA kernels.

## Why

I wanted to understand what actually happens when you call `loss.backward()` in PyTorch.

Not at the level of "it computes gradients using the chain rule" but at the level of: 
what data structure is the computation graph? How does a tensor know which operation 
created it? Why does PyTorch use `shared_ptr` internally? What is a CUDA kernel actually 
doing when matmul runs on a GPU, and why is it 17x faster than a CPU at 1024x1024 but 
slower at 64x64?

The goal of this project isn't to replace production libraries, but to engineer the core of a tensor engine from first principles. It is an exploration of C++ memory management, PCIe bottlenecking, parallel hardware architecture, and topological graph traversal. I wanted to build the tools, understand the hardware limitations, and run a real model with them.

Along the way I hit the bugs that explain the design decisions: the dangling reference problem that makes `shared_ptr` non-negotiable, the `std::vector` allocation overhead that made tiling appear slower than naive, the loop reordering that beat manual cache blocking by 2x on CPU while being useless on GPU

## What is implemented

**Tensor core**
* N-dimensional tensor backed by a flat `std::vector<float>`, with shape and stride-based indexing
* Stride computation supports arbitrary dimensionality
* Device abstraction with `CPU` and `CUDA`
* Tensor transfers via `Tensor::cpu()` and `Tensor::cuda()`
* GPU tensor storage using device memory (`gpu_data`)

**Matrix multiplication**
* Naive $O(n^3)$ implementation (CPU)
* Cache-blocked (tiled) matrix multiplication (CPU)
* Loop-reordered `i-k-j` implementation (CPU)
* Naive CUDA implementation (2D Grid/Block mapping)
* Tiled CUDA implementation (Shared memory, cooperative loading, `__syncthreads()`)
* Automatic CPU/CUDA dispatch based on tensor device

**Tensor operations**
* `add`, `multiply`, `relu`, `scale`, `transpose`, `sum`, `softmax`
* CUDA implementations for `add`, `multiply`, `scale`, `transpose`, `relu`, and `softmax`
* Shared-memory reduction kernel for row-wise softmax

**Autograd engine**
* Reverse-mode automatic differentiation over a dynamically built computation graph
* Each tensor stores `shared_ptr` references to its inputs (`_prev`), keeping the graph alive for backward passes
* `backward()` performs a topological sort and calls each node's `backward_fn` in reverse order
* Gradients implemented for matmul, add, multiply, relu, transpose, and sum
* CPU autograd support with forward-only CUDA execution

**Training demo**
* A 2-layer MLP trained on XOR using only this library
* Forward pass: `matmul -> add -> relu -> matmul -> add -> MSE loss`
* Trained with SGD implemented from scratch
* Converges from loss ~1.68 to ~0 within 100 epochs
* Exercises the full autograd engine, including forward propagation, backward propagation, and parameter updates

**Transformer Components**
* Single-head attention forward pass (`Q @ K^T / sqrt(d_k) → softmax → @ V`)
* CPU and GPU implementations verified to match within 1e-3 relative error

## Benchmarks

### 1. CPU Matmul Scaling (Naive vs Tiled, -O2)

| Matrix Size (N x N) | Naive (ms) | Tiled (ms) | Speedup | 
| ----- | ----- | ----- | ----- | 
| 64 | 0.32 | 0.37 | 0.86x | 
| 128 | 2.50 | 2.54 | 0.98x | 
| 256 | 30.05 | 19.54 | 1.53x | 
| 512 | 245.92 | 199.34 | 1.23x | 
| 1024 | 2644.25 | 2232.56 | 1.18x | 

*Note: The initial tiled implementation was not faster than naive due to overhead in the generic `at()` indexing function (a `std::vector` allocation on every element access). Switching to a direct-index `at2d()` and using tile size 32 produced the numbers above. This was a useful lesson in measuring rather than assuming where bottlenecks are. Measured with i7-9750H CPU on WSL2.*

### 2. CPU Matrix Multiplication Optimization (-O3, i7-9750H, 512x512 matrix)

Optimization is often counter-intuitive. Below is the journey of improving CPU matrix multiplication performance through loop reordering and cache-aware memory access patterns.

| Implementation | Execution Time | The "Why" |
| ----- | ----- | ----- |
| **Naive (3 Loops `i-j-k`)**<br>Standard row/col iteration. | ~245.92 ms | Constant cache misses jumping down columns of B. |
| **Cache-Blocked (6 Loops `i-j-k`)**<br>Outer loops for tiles, inner loops for elements. | ~199.34 ms | Manual Cache Blocking (Tiling) forces data to stay in L1 cache. |
| **Cache-Blocked (6 Loops `i-k-j`)**<br>Tiled with inner memory sequential. | ~24.20 ms | Swapping inner loops ensures sequential memory access for AVX vectorization. |
| **Naive (3 Loops `i-k-j`)**<br>Standard iteration, sequential inner memory. | **~12.13 ms** | **Fastest.** With memory strictly sequential, the CPU's hardware prefetcher handles the cache perfectly. Manual tiling just adds loop overhead! |

### 3. Final Hardware Scaling (CPU `i-k-j` vs CUDA Tiled)

This table demonstrates the fundamental hardware architecture differences between a mobile CPU (i7-9750H) and a discrete GPU (RTX 2060).

| Size | Optimized CPU (ms) | Tiled GPU Kernel (ms) | Speedup |
| ----- | ----- | ----- | ----- |
| 64 | 0.01 | 0.06 | 0.24x |
| 128 | 0.12 | 0.12 | 1.00x |
| 256 | 1.17 | 0.13 | 9.03x |
| 512 | 8.98 | 0.69 | 13.09x |
| 1024 | 81.81 | 4.61 | 17.74x |
| 2048 | 1408.78 | 47.77 | 29.49x |


*Note: Results are averaged over 10 runs after a warm-up execution. At small matrix sizes, kernel launch overhead dominates and the CPU can be competitive. As matrix size grows, the GPU's parallel execution model and shared memory tiling provide increasingly large speedups over the optimized CPU implementation.*

## Design notes

**Deep Dive into Hardware Optimizations**
For a detailed breakdown of the systems engineering lessons learned while writing the backend kernels (including memory coalescing, shared memory tiling, cache blocking, and the "-O3 compiler illusion"), please read the extensive notes in [`cuda/NOTES.md`](cuda/NOTES.md).

**Why `shared_ptr` everywhere**
Backward functions need to reference the tensors that produced them, but those tensors may go out of scope before `backward()` is called. Each tensor holds `shared_ptr` references to its inputs (`_prev`), which keeps the entire computation graph alive for as long as the output tensor is alive. This is the exact approach PyTorch uses internally (`shared_ptr<TensorImpl>`).

**Raw vs autograd-tracked ops**
Internal operations (`matmul_raw`, `transpose_raw`, etc.) perform pure computation with no graph tracking. The public versions (`matmul`, `transpose`, etc.) wrap these and additionally set `_prev` and `backward_fn`. This separation prevents backward passes from accidentally building new graph nodes during gradient computation.

## Build & run

```bash
# Build (CPU + CUDA)
mkdir build && cd build
cmake ..
make -j4

# Build (CPU only)
mkdir build && cd build
cmake .. -DSKIP_CUDA=ON
make -j4

# Run tests
./build/test_tensor
./build/xor
./build/test_gradcheck
./build/test_parity    # GPU only
./build/test_dispatch  # GPU only
```

## Roadmap

### Core Tensor Library
- [x] Tensor struct with shape and strides
- [x] Elementwise ops (add, multiply, relu, transpose, sum, scale)
- [ ] Broadcasting
- [ ] Tensor slicing and views
- [ ] Serialization

### Autograd
- [x] Reverse-mode autograd engine
- [x] Topological graph traversal
- [x] Gradient checking via finite differences
- [x] Softmax
- [ ] Additional ops (GELU, LayerNorm)

### CPU Backend
- [x] Naive matmul
- [x] Cache-blocked matmul
- [x] Loop-order optimization (`i-k-j`)
- [x] CPU benchmarking

### CUDA Backend
- [x] Vector add kernel
- [x] Naive CUDA matmul
- [x] Shared-memory tiled matmul
- [x] CPU vs GPU benchmarks
- [x] CUDA elementwise kernels (add, multiply, relu, scale)
- [x] CUDA softmax kernel (parallel reduction)
- [x] CUDA transpose kernel
- [x] Device abstraction (`CPU` / `CUDA`)
- [x] GPU tensor storage

### Transformer Components
- [x] Single-head attention forward pass (CPU + GPU verified)
- [ ] Broadcasting for bias addition
- [ ] Multi-head attention
- [ ] One full transformer block (attention + LayerNorm + FFN)
- [ ] Load pretrained weights and run inference

### Neural Network Components
- [x] XOR MLP demo
- [x] SGD optimizer
- [ ] Linear layer abstraction
- [ ] Optimizer API (Adam, AdamW)
- [ ] MLP training on real dataset

### Long-Term
- [ ] Caching memory allocator
- [ ] cuBLAS backend
- [ ] Full GPU autograd