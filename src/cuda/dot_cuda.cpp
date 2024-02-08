#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> dot_cuda_forward(
    torch::Tensor input_left,
    torch::Tensor input_right);

std::vector<torch::Tensor> dot_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor input_right);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> dot_forward(
    torch::Tensor input_left,
    torch::Tensor input_right)
{
  CHECK_INPUT(input_left);
  CHECK_INPUT(input_right);

  return dot_cuda_forward(input_left, input_right);
}

std::vector<torch::Tensor> dot_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor input_right)
{
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input_left);
  CHECK_INPUT(input_right);

  return dot_cuda_backward(
      grad_h,
      input_left,
      input_right);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &dot_forward, "dot forward (CUDA)");
  m.def("backward", &dot_backward, "dot backward (CUDA)");
}