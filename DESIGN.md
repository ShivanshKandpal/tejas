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

## 6. Gradient Verification, Why finite differences

Deriving gradients by hand is error-prone, especially for operations such as softmax and LayerNorm where every output depends on every input. Rather than trusting manual derivations, every differentiable operation in tejas is verified using finite-difference gradient checking. Numerical gradients are computed independently of the autograd engine and compared against analytical gradients produced during backpropagation. This caught several subtle bugs during development, including mistakes in broadcasting and an incorrect LayerNorm test where the chosen objective function was identically zero.

## 7. Why cache Intermediate Values in Layernorm Backward

LayerNorm's backward pass depends on the normalized activations (`x_hat`) and the inverse standard deviation (`inv_std`) computed during the forward pass. While both quantities could be recomputed during backpropagation, doing so would repeat work and complicate the implementation. Instead, tejas caches them during the forward pass and captures them inside the backward lambda.

Unlike trainable tensors, these cached values never require gradients and are not part of the computation graph. They are therefore stored as `shared_ptr<std::vector<float>>` rather than `Tensor` objects, avoiding unnecessary graph nodes while still ensuring the data remains alive until `backward()` is called.

The backward implementation follows the fused LayerNorm formulation used by modern deep learning frameworks. Rather than constructing the full Jacobian of the normalization operation, it first computes two row-wise reductions (`Σ grad_xhat` and `Σ grad_xhat · x_hat`) before performing a final linear pass over the row to compute the input gradients. This keeps the implementation efficient while remaining mathematically equivalent to the full derivative.

## 8. Why make Layer Abstractions (Neural Network Modules)

Primitive tensor operations (`matmul`, `add`, `layernorm`, etc.) are responsible for computation and autograd, while higher-level modules such as `Linear` and `LayerNorm` simply own trainable parameters and compose those operations. This separation keeps tensor operations reusable while providing an interface familiar to users of modern deep learning frameworks. Modules also expose a common `parameters()` interface and support in-place device transfers through `.cpu()` and `.cuda()`, allowing optimizers and training loops to operate on layers without knowledge of their internal implementation.

## 9. Why Attention Has an Output Projection

Single-head attention does not return the final block output directly. After attention, the tensor has shape `[seq_len, d_k]`, but the residual path expects `[seq_len, d_model]`. The output projection `o_proj` maps the attention result back to `d_model`, which makes the residual connection work cleanly and also allows `d_k` to differ from `d_model`. This matches the original transformer design and keeps the attention module flexible.

---

## 10. Why FeedForward Is Its Own Module

The feed-forward network is a separate module instead of being baked into `TransformerBlock` directly. That keeps the block small and consistent with the rest of the codebase, where reusable pieces own their parameters and expose a `forward()` method. It also makes the FFN easy to evolve independently, for example if dropout, gated variants, or a different activation are added later.

---

## 11. Why Pre-Norm Instead of Post-Norm

The transformer block uses the pre-norm layout, where LayerNorm happens before attention and before the feed-forward network. That keeps the residual stream untouched, which usually makes optimization easier and gradient flow cleaner. It also matches the style used by a lot of modern transformer models, including GPT-style architectures.
