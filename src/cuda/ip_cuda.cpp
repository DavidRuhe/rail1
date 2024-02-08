#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> ip_cuda_forward(
    torch::Tensor input_left,
    torch::Tensor metric,
    torch::Tensor input_right);

std::vector<torch::Tensor> ip_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor metric,
    torch::Tensor input_right);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ip_forward(
    torch::Tensor input_left,
    torch::Tensor metric,
    torch::Tensor input_right)
{
  CHECK_INPUT(input_left);
  CHECK_INPUT(metric);
  CHECK_INPUT(input_right);

  return ip_cuda_forward(input_left, metric, input_right);
}

std::vector<torch::Tensor> ip_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor metric,
    torch::Tensor input_right)
{
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input_left);
  CHECK_INPUT(metric);
  CHECK_INPUT(input_right);

  return ip_cuda_backward(
      grad_h,
      input_left,
      metric,
      input_right);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &ip_forward, "ip forward (CUDA)");
  m.def("backward", &ip_backward, "ip backward (CUDA)");
}