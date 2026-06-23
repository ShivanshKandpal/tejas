# Design Notes

Technical decisions behind tejas and the reasoning that led to them.

---

## 1. Why `shared_ptr`

My first autograd implementation captured tensors by reference inside backward lambdas. It worked for simple tests, but eventually I started running into dangling references.

For example, if `matmul(a, b)` creates a tensor `c`, the local `c` inside `matmul` goes out of scope when the function returns. Any backward lambda holding a reference to it is now pointing at invalid memory.

The solution was to allocate tensors with `shared_ptr` and store `shared_ptr` references to parent tensors in `_prev`. As long as the output tensor exists, the tensors needed for backward also stay alive.

This is also the general approach used by PyTorch internally.

---

## 2. Why `_raw` Functions exist

Operations such as `matmul`, `add`, and `relu` do more than just compute values. They also build the autograd graph by setting `_prev` and `backward_fn`.

During backward, I only want the actual computation, not more graph construction. If a backward function called `matmul` instead of `matmul_raw`, it would create new graph nodes while traversing the graph, which quickly becomes a mess.

To avoid this, every operation has a `_raw` version that performs only the computation. The public function wraps the raw version and adds autograd bookkeeping on top.

This separation also makes backend dispatch cleaner because the raw functions can choose between CPU and CUDA implementations without affecting the autograd graph.

---

## 3. Why Operations Take `const TensorPtr&`

Most operations take tensors as `const TensorPtr&` instead of by value.

`std::shared_ptr` maintains an atomic reference count, so copying it means updating that counter. The cost is small, but tensor operations get called constantly.

Passing by reference avoids unnecessary reference count updates during the forward pass. When a tensor actually needs to be kept alive for autograd, the backward lambda captures the `shared_ptr` by value, which is where the extra reference is genuinely needed.

The result is slightly less overhead during inference while still preserving correct graph lifetimes during training.

---

## 4. Why CUDA Kernels Map `threadIdx.x` to Columns

CUDA executes threads in groups of 32 called warps. Consecutive values of `threadIdx.x` belong to the same warp.

Since tensors are stored in row-major order, consecutive columns in a row are stored next to each other in memory. Mapping `threadIdx.x` to the column index means neighboring threads access neighboring memory locations.

This allows the GPU to combine memory requests into larger transactions, which is much faster than having each thread read from unrelated locations.

Because of this, kernels in tejas generally follow the convention:

`row = blockIdx.y * blockDim.y + threadIdx.y;`
`col = blockIdx.x * blockDim.x + threadIdx.x;`

---

## 5. Why `i-k-j` Ended up Faster Than Tiling on CPU

The original CPU matmul used the standard `i-j-k` loop order. Performance was poor because the innermost loop walks down columns of `B`, which leads to a large number of cache misses.

I then implemented cache blocking (tiling), expecting it to be the main optimization. It helped, but the improvement was smaller than I expected.

The biggest gain came from changing the loop order to `i-k-j`.

By loading `A[i][k]` once and making `j` the innermost loop, accesses to both `B` and `C` become sequential. This greatly improves cache locality and also gives the compiler a much better opportunity to vectorize the loop.

For a 512×512 matrix on my i7-9750H, this reduced execution time from roughly 245 ms to around 12 ms.

One of the more surprising lessons from this project was that on a modern CPU, fixing the memory access pattern mattered far more than manually managing the cache.