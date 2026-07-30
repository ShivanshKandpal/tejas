# Systems Programming & CUDA Notes

## 1. Vector Add: Memory-Bound Workload

Benchmark on RTX 2060, n = 1,000,000 floats:

| | Time |
|---|---|
| CPU (single loop) | 6.09 ms |
| GPU kernel only | 2.09 ms |
| GPU total (malloc + memcpy + kernel + memcpy) | 330.79 ms |

**Kernel-only speedup: 2.9x. Total speedup: 0.018x (54x slower).**

### Why
Vector addition does one FLOP per element (`c[i] = a[i] + b[i]`) but requires moving 3 floats (read a, read b, write c) per element. The amount of work per byte moved is tiny, making this a heavily **memory-bound** operation.

`cudaMalloc` and `cudaMemcpy` have significant fixed overhead (driver calls, PCIe transfer). For a workload this cheap, that overhead dwarfs the actual computation. The GPU genuinely computes faster (2.9x), but getting the data there and back costs far more than the computation saves.

### Implication
GPUs win when there is a high ratio of **compute per byte transferred** (Arithmetic Intensity). For example, matmul of an NxN matrix ($O(N^2)$ data) requires $O(N^3)$ operations. The compute-to-data ratio grows with N, so transfer overhead becomes negligible relative to compute.

---

## 2. Naive Matmul: Compute-Bound (with Latency Bottlenecks)

### The "-O3 Compiler Illusion"

Without `-O3` and `-march=native`, the CPU loop executes unvectorized and poorly optimized, producing misleading comparisons against the GPU.

Compiler optimizations alone reduced CPU execution time by an order of magnitude. Later, changing the loop order from `i-j-k` to `i-k-j` improved cache locality and vectorization further, reducing execution time to ~12.13 ms for a 512x512 matrix on an i7-9750H.

### Memory Coalescing (`threadIdx.x` mapped to columns)
In the 2D CUDA grid, threads are grouped into Warps of 32 along the `threadIdx.x` dimension.
* By mapping `x` to columns (`col = blockIdx.x * blockDim.x + threadIdx.x`), adjacent threads read adjacent memory addresses in the row-major flattened array. 
* The GPU memory controller bundles these 32 contiguous reads into a single transaction (**Memory Coalescing**), maximizing bandwidth. Swapping `x` to `row` results in strided accesses, crushing performance.

### The Global Memory Bottleneck
While naive execution is fast, it forces threads to fetch Matrix A and B entirely from slow global VRAM. For a 512x512 matrix, data from A and B is repeatedly fetched from global memory by many threads, creating significant memory traffic and limiting performance.

---

## 3. Tiled Matmul: Shared Memory Optimization (GPU)

To overcome the global memory read bottleneck, we utilize `__shared__` memory. This is a programmer-managed memory space located on the Streaming Multiprocessor (SM) with much lower latency than global memory.

| Implementation (512x512) | GPU Kernel Time | Speedup vs Naive GPU |
|---|---|---|
| Naive Matmul | ~1.95 ms | 1.0x |
| **Tiled Matmul** | **~0.69 ms** | **2.8x** |

### The Cooperative Loading Mechanism
1. **Cooperative Load:** Instead of each thread fetching a full row/col independently, a block of threads (e.g., 16x16) acts as a team. Each thread loads exactly *one* float into the `__shared__` memory cache.
2. **Barrier Synchronization:** `__syncthreads()` acts as a physical hardware barrier. No thread can proceed to the math phase until the entire 16x16 tile is successfully loaded into the cache.
3. **Fast Math:** Threads compute the partial dot product using the shared memory, entirely bypassing global VRAM latency (`#pragma unroll` is used here for maximum instruction throughput).
4. **Slide and Repeat:** The block calls `__syncthreads()` again to prevent overwriting, slides the tile window down the matrix, and repeats.

---

## 4. CPU Matmul: Hardware Prefetcher vs Cache Blocking

To create a perfectly fair hardware comparison, the CPU code must also be optimized. We tested standard algorithms against manual Cache Blocking.

* **3-Loop (Naive):** The standard mathematical definition of matrix multiplication (iterating through rows and columns).
* **6-Loop (Cache-Blocked/Tiled):** Dividing the matrices into smaller blocks. The 3 outer loops iterate over the tiles, and the 3 inner loops perform standard matrix multiplication strictly within the bounds of a single tile to keep it in the L1 cache.

Benchmark on i7-9750H, $M=N=K=512$:

| CPU Implementation | Execution Time | Notes |
|---|---|---|
| Naive (3 Loops `i-j-k`) | ~245.92 ms | Standard memory striding. Horrible cache performance. |
| Cache-Blocked (6 Loops `i-j-k`) | ~199.34 ms | Manual Cache Blocking. Forces blocks to stay in L1 cache. |
| Cache-Blocked (6 Loops `i-k-j`) | ~24.20 ms | Cache Blocking + inner loop memory sequentially aligned. |
| **Naive (3 Loops `i-k-j`)** | **~12.13 ms** | **Fastest:** Pure sequential memory access. |

### The Core Lesson: CPU vs GPU Caching

While manual cache blocking (tiling) is essential on the GPU to utilize `__shared__` memory efficiently, it actually hurt performance on the CPU once the memory access pattern was fixed.

By pulling `A[i][k]` into a register and making `j` the innermost loop:

```cpp
for(int i = 0; i < M; i++){
    for(int k = 0; k < K; k++){
        float a_ik = h_A[i * K + k];
        for(int j = 0; j < N; j++){
            h_C_cpu[i * N + j] += a_ik * h_B[k * N + j];
        }
    }
}
```

The biggest CPU improvement did not come from cache blocking. It came from changing the loop order.

The original implementation used:

```cpp
for(i)
    for(j)
        for(k)
```

which walks down columns of `B` and causes poor cache utilization.

Changing the order to:

```cpp
for(i)
    for(k)
        for(j)
```

allows both `B` and `C` to be accessed sequentially in memory. This dramatically improves cache locality and enables aggressive compiler vectorization, reducing execution time from ~245 ms to ~12 ms for a 512x512 matrix on an i7-9750H.

The result was surprising: a simple 3-loop `i-k-j` implementation outperformed the manually cache-blocked 6-loop version. On this CPU, the hardware prefetcher and compiler optimizations were more effective than explicit cache blocking.

## Key Takeaways

1. GPUs only outperform CPUs when arithmetic intensity is high enough to amortize transfer and launch overhead.
2. Memory access patterns matter more than arithmetic. Reordering loops improved CPU performance by roughly 20x without changing the algorithm.
3. Shared memory tiling is essential on GPUs because global memory bandwidth becomes the bottleneck.
4. Benchmarking optimized CPU code is critical. Comparing against unoptimized code produces misleading speedup numbers.

## Final Hardware Comparison

Average of 10 runs after warm-up.

| Size | CPU (i-k-j) | GPU Tiled | Speedup |
|---|---|---|---|
| 64 | 0.01 ms | 0.06 ms | 0.24x |
| 128 | 0.12 ms | 0.12 ms | 1.00x |
| 256 | 1.17 ms | 0.13 ms | 9.03x |
| 512 | 8.98 ms | 0.69 ms | 13.09x |
| 1024 | 81.81 ms | 4.61 ms | 17.74x |
| 2048 | 1408.78 ms | 47.77 ms | 29.49x |

## 5. Parallel Reduction: Softmax on the GPU

Elementwise operations such as Add and ReLU are straightforward to parallelize because each thread can process one element independently. Softmax is different because it requires information from the entire row.

For each row, we need to:

1. Find the maximum value (for numerical stability).
2. Compute exponentials.
3. Sum the exponentials.
4. Normalize each element by the sum.

Finding the maximum and computing the sum are reduction operations, which require cooperation between threads in a block.

### Tree Reduction

A naive approach would have a single thread scan the entire row to compute the maximum or sum. This works, but it leaves most of the GPU idle.

Instead, we use a tree reduction in shared memory. At each step, half of the threads combine their values with another element a fixed distance away. The distance (`stride`) is halved every iteration until a single value remains.

For example:

```text
8 values
↓
4 partial results
↓
2 partial results
↓
1 final result
```

This reduces the amount of work per thread and allows the entire block to participate in the reduction.

### Why `__syncthreads()` Is Needed

During the reduction, threads repeatedly read values written by other threads in previous iterations.

```cpp
for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if(threadIdx.x < stride) {
        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
}
```

The synchronization barrier ensures that all writes for the current iteration are visible before the next iteration begins. Without it, some threads could read values that have not been updated yet, producing incorrect results.

### Handling Non-Power-of-Two Sizes

The reduction above works most naturally when the number of threads is a power of two.

For example:

```text
2, 4, 8, 16, 32, 64, ...
```

Real tensor dimensions are often not powers of two. A row might contain 13 elements instead of 16.

To handle this, we launch the next power-of-two number of threads and pad the extra entries in shared memory.

For a max reduction:

```cpp
sdata[tid] = -CUDART_INF_F;
```

For a sum reduction:

```cpp
sdata[tid] = 0.0f;
```

These values do not affect the final result, allowing the same reduction code to work for arbitrary row sizes while keeping the reduction tree balanced.

## 6. Transpose Kernel: Coalescing Trade-off

Naive GPU transpose has an inherent coalescing problem, either the read 
or the write will be strided. With x→columns mapping, reads from input 
are coalesced but writes to output (col*rows + row) are strided.

The optimized solution uses shared memory tiling: load a tile coalesced 
into shared memory, then write coalesced from shared memory to output. 
I have planned this as future optimization, my reasoning is that current naive kernel is correct and fast enough for attention dimensions (d_k ≤ 128), where the strided
write overhead is negligible compared to the matmul kernels surrounding it.


## 7. Attention: Putting It All Together

Single-head attention exercises every kernel in the library in sequence:
```
input → matmul(W_q) → Q  [seq_len, d_k]
input → matmul(W_k) → K  [seq_len, d_k]
input → matmul(W_v) → V  [seq_len, d_k]

scores = matmul(Q, transpose(K))     [seq_len, seq_len]
scores = scale(scores, 1/sqrt(d_k))  [seq_len, seq_len]
attn   = softmax(scores)             [seq_len, seq_len]
output = matmul(attn, V)             [seq_len, d_k]
```
CPU and GPU implementations produce identical results within 1e-3 
relative error, verified by hardware parity test on RTX 2060.

## 8. LayerNorm: Row-wise Normalization and Fused Backward

LayerNorm normalizes each row independently. For each row, it computes the mean and variance, then rescales the values so that the normalized activations have approximately zero mean and unit variance before applying the learnable scale (`gamma`) and shift (`beta`).

The backward pass is where most of the complexity lies. A direct derivation produces a dense Jacobian because every output depends on every input through both the mean and the variance. Instead of explicitly constructing that Jacobian, the implementation uses the standard fused LayerNorm backward formula:

$$ \frac{\partial L}{\partial x} = \frac{\text{inv\_std}}{N} \left( N \cdot \text{grad}_{\hat{x}} - \sum \text{grad}_{\hat{x}} - \hat{x} \cdot \sum(\text{grad}_{\hat{x}}\hat{x}) \right) $$

The implementation caches `x_hat` and `inv_std` during the forward pass so they do not need to be recomputed during backpropagation. This reduces the backward pass to two row-wise reductions followed by one final linear pass over the row.

---

## 9. Gradient Checking: Choosing the Right Objective

One of the first LayerNorm gradient checks I wrote used the following objective:

```cpp
sum(layernorm(x, gamma, beta))
```

with `gamma = 1` and `beta = 0`.

This turned out to be a poor test. Since LayerNorm produces normalized outputs with approximately zero mean for each row, the summed output becomes nearly constant. The analytical gradient is therefore almost zero, while the numerical gradient is dominated by floating-point error, producing misleading mismatches.

Changing the objective to a weighted sum of the normalized outputs produced meaningful gradients and correctly verified the backward implementation. It was a useful reminder that finite-difference gradient checking depends just as much on the chosen objective as it does on the correctness of the backward pass.

---

## 10. GELU: Choosing the Sigmoid Approximation

The original GELU activation is defined using the Gaussian cumulative distribution function, which is relatively expensive to compute. Instead, tejas uses the simpler sigmoid approximation:

$$

\mathrm{GELU}(x)
\approx
x \cdot \sigma(1.702x)

$$

This approximation is inexpensive to compute, easy to differentiate, and keeps both the forward and backward implementations compact. While it is not identical to the exact GELU or the commonly used tanh approximation, it is sufficiently accurate for an educational deep learning framework while keeping the implementation straightforward.

## 11. Building the First Transformer Block

The transformer block itself did not require any new tensor operations. Instead, it was built by composing the modules that already existed: `LayerNorm`, `SingleHeadAttention`, `FeedForward`, and residual connections. Most of the work ended up being debugging how those pieces fit together correctly.

### Bug 1: Attention Output Shape

My first implementation of `SingleHeadAttention` returned the attention output directly:

```text
Attention(Q, K, V) -> [seq_len, d_k]
```

This worked as long as `d_k == d_model`, but immediately broke the residual connection when I tried different dimensions:

```cpp
add(x, attention_out);
```

since

```text
x              : [seq_len, d_model]
attention_out  : [seq_len, d_k]
```

The fix was to add an output projection:

```text
output = W_o(Attention(Q, K, V))

W_o : [d_k, d_model]
```

which maps the attention output back to the model dimension. This is the same role played by the `W^O` projection in the original Transformer's multi-head attention block. With the output projection in place, the residual connection works regardless of the chosen head dimension.

### Bug 2: Residual Connection

I also made a subtle mistake in the second residual connection. My initial implementation was

```cpp
add(x, ffn_out);
```

This is incorrect because the feed-forward network should operate on the output of the attention block, not on the original input.

The correct implementation is

```cpp
out1 = add(x, attn_out);
out2 = add(out1, ffn_out);
```

The bug was easy to miss because the code compiled and produced outputs, but the computation graph no longer matched the Transformer architecture.

### Gradient Checking the Entire Block

After wiring all of the modules together, I verified the complete transformer block using finite-difference gradient checking.

Rather than checking attention, LayerNorm, and the feed-forward network independently, gradients were propagated through the entire computation graph:

```text
LayerNorm
    ↓
Single-Head Attention
    ↓
Residual
    ↓
LayerNorm
    ↓
Feed-Forward Network
    ↓
Residual
```

The gradient check is noticeably slower than checking individual operators because each numerical gradient requires additional forward evaluations, but it provides strong confidence that all of the modules compose correctly into a single computation graph.

### Parameter Count

For the test configuration

```text
d_model = 8
d_k     = 4
d_ff    = 32
```

the transformer block contains:

| Module | Parameters |
|--------|-----------:|
| Q projection | 36 |
| K projection | 36 |
| V projection | 36 |
| Output projection | 40 |
| LayerNorm 1 | 16 |
| FeedForward | 552 |
| LayerNorm 2 | 16 |
| **Total** | **732** |

Breaking the parameter count down by module gave me a better intuition for where the model capacity actually comes from. Even in this tiny example, the feed-forward network contains far more parameters than the attention mechanism itself.