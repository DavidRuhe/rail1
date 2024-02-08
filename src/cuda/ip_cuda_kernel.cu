#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <torch/types.h>

#include <vector>
#include <torch/torch.h>

// Assuming BLOCK_SIZE is defined somewhere
#define BLOCK_SIZE 256

__global__ void ip_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> input_left,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> metric,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> output,
    const size_t N, const size_t M)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int colIndex = blockIdx.x;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Placeholder for if index >= N
    float sum = 0;

    // Each thread computes its product
    if (index < N && colIndex < M && metric[index][colIndex] != 0)
    {
        sum = input_left[index] * metric[index][colIndex]
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
        atomicAdd(&output[colIndex], sdata[0]);
    }
}

std::vector<torch::Tensor> ip_cuda_forward(
    torch::Tensor input_left,
    torch::Tensor metric, )
{
    const auto size = input_left.size(0);
    const auto M = metric.size(1);
    const int threads = BLOCK_SIZE;
    const int num_blocks = (size + threads - 1) / threads;
    auto output = torch::zeros({M}, input_left.options());

    // Calculate shared memory size per block
    const size_t shared_memory_size = threads * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(input_left.type(), "ip_forward_cuda", ([&]
                                                                      { ip_forward_modified_cuda_kernel<<<num_blocks, threads, shared_memory_size>>>(
                                                                            input_left.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                            metric.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                                                            output.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                                            N, M); }));

    return {output};
}

template <typename scalar_t>
__global__ void ip_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_left,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> grad_result,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> metric,
    const size_t N, const size_t M)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N)
    {
        scalar_t gradient = 0;
        for (int col = 0; col < M; ++col)
        {
            // Accumulate gradient contributions from all columns of metric
            gradient += grad_result[col] * metric[row][col];
        }
        d_left[row] = gradient;
    }
}
std::vector<torch::Tensor> ip_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor metric,
    torch::Tensor input_right)
{
    auto d_left = torch::zeros_like(input_left);
    auto d_right = torch::zeros_like(input_right);

    const auto size = input_left.size(0);
    const int threads = 1024;
    const dim3 blocks((size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(grad_h.type(), "ip_backward_cuda", ([&]
                                                                   { ip_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                                                         d_left.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                         d_right.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                         grad_h.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                         input_left.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                         metric.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                                                         input_right.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()); }));

    return {d_left, d_right};
}
