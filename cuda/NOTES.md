# CUDA Notes

## Vector Add — Memory-bound workload

Benchmark on RTX 2060, n = 1,000,000 floats:

| | Time |
|---|---|
| CPU (single loop) | 6.09 ms |
| GPU kernel only | 2.09 ms |
| GPU total (malloc + memcpy + kernel + memcpy) | 330.79 ms |

**Kernel-only speedup: 2.9x. Total speedup: 0.018x (54x slower).**

### Why

Vector addition does one FLOP per element (`c[i] = a[i] + b[i]`) but requires
moving 3 floats (read a, read b, write c) per element. The amount of work per
byte moved is tiny and this is a **memory-bound** operation.

`cudaMalloc` and `cudaMemcpy` have significant fixed overhead (driver calls,
PCIe transfer). For a workload this cheap, that overhead dwarfs the actual
computation. The GPU genuinely computes faster (2.9x), but getting the data
there and back costs far more than the computation saves.

### Implication

GPUs win when there's a lot of **compute per byte transferred** e.g. matmul,
where an NxN matrix (O(N²) data) requires O(N³) operations. The compute-to-data
ratio grows with N, so transfer overhead becomes negligible relative to compute.
This is also why real systems keep tensors resident on GPU across many
operations rather than copying back and forth, the cost of `cudaMalloc`/
`cudaMemcpy` is amortized once, not paid per-op.

I am expecting matmul to tell a very different story.