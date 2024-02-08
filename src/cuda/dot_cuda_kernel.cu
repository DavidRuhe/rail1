#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <torch/types.h>

#include <vector>
#include <torch/torch.h>

// Assuming BLOCK_SIZE is defined somewhere
#define BLOCK_SIZE 1024

__global__ void dot_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> input_left,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> input_right,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> output,
    const size_t N)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;

    // Each thread computes its product
    if (index < N)
    {
        sum = input_left[index] * input_right[index];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

std::vector<torch::Tensor> dot_cuda_forward(
    torch::Tensor input_left,
    torch::Tensor input_right)
{
    const auto size = input_left.size(0);
    const int threads = BLOCK_SIZE;
    const int num_blocks = (size + threads - 1) / threads;
    auto partial_sums = torch::empty({num_blocks}, input_left.options());

    // Calculate shared memory size per block
    const size_t shared_memory_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input_left.type(), "dot_forward_cuda", ([&]
                                                                       { dot_forward_cuda_kernel<<<num_blocks, threads, shared_memory_size>>>(
                                                                             input_left.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                             input_right.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                             partial_sums.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                             size); }));

    auto total_sum = partial_sums.sum();

    return {total_sum}; // Return the total sum as the result of the dot product
}

template <typename scalar_t>
__global__ void dot_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_left,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_right,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_result,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input_left,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input_right)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d_left.size(0))
    {
        d_left[c] = grad_result[0] * input_right[c];
        d_right[c] = grad_result[0] * input_left[c];
    }
}

std::vector<torch::Tensor> dot_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor input_right)
{
    auto d_left = torch::zeros_like(input_left);
    auto d_right = torch::zeros_like(input_right);

    const auto size = input_left.size(0);
    const int threads = 1024;
    const dim3 blocks((size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(grad_h.type(), "dot_backward_cuda", ([&]
                                                                    { dot_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                                                          d_left.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                          d_right.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                          grad_h.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                          input_left.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                          input_right.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()); }));

    return {d_left, d_right};
}
