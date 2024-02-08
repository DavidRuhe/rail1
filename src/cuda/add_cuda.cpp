#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> add_cuda_forward(
    torch::Tensor input_left,
    torch::Tensor input_right);

std::vector<torch::Tensor> add_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor input_right
    // torch::Tensor grad_right,
    // torch::Tensor new_cell,
    // torch::Tensor input_gate,
    // torch::Tensor output_gate,
    // torch::Tensor candidate_cell,
    // torch::Tensor X,
    // torch::Tensor gate_weights,
    // torch::Tensor weights
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> add_forward(
    torch::Tensor input_left,
    torch::Tensor input_right)
{
  CHECK_INPUT(input_left);
  CHECK_INPUT(input_right);

  return add_cuda_forward(input_left, input_right);
}

std::vector<torch::Tensor> add_backward(
    torch::Tensor grad_h,
    torch::Tensor input_left,
    torch::Tensor input_right
    // torch::Tensor grad_cell,
    // torch::Tensor new_cell,
    // torch::Tensor input_gate,
    // torch::Tensor output_gate,
    // torch::Tensor candidate_cell,
    // torch::Tensor X,
    // torch::Tensor gate_weights,
    // torch::Tensor weights
)
{
  CHECK_INPUT(grad_h);
  CHECK_INPUT(input_left);
  CHECK_INPUT(input_right);
  // CHECK_INPUT(grad_cell);
  // CHECK_INPUT(input_gate);
  // CHECK_INPUT(output_gate);
  // CHECK_INPUT(candidate_cell);
  // CHECK_INPUT(X);
  // CHECK_INPUT(gate_weights);
  // CHECK_INPUT(weights);

  return add_cuda_backward(
      grad_h,
      input_left,
      input_right
      // grad_cell,
      // new_cell,
      // input_gate,
      // output_gate,
      // candidate_cell,
      // X,
      // gate_weights,
      // weights
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &add_forward, "Add forward (CUDA)");
  m.def("backward", &add_backward, "Add backward (CUDA)");
}