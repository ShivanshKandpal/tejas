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