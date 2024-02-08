import torch
from torch import nn
import dot_cuda


class DotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_left, input_right):
        outputs = dot_cuda.forward(input_left, input_right)
        new_h = outputs[0]
        # variables = outputs[1:] + [weights]
        variables = [input_left, input_right]
        ctx.save_for_backward(*variables)
        return new_h

    @staticmethod
    def backward(ctx, grad_h):
        grad_input_left, grad_input_right = dot_cuda.backward(
            grad_h[None].contiguous(), *ctx.saved_tensors
        )
        return grad_input_left, grad_input_right


class Dot(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.input_left = nn.Parameter(torch.arange(size).float() + 10)
        self.input_right = nn.Parameter(torch.arange(size).float())

        # self.input_left = torch.arange(size).float().cuda()
        # self.input_right = torch.arange(size).float().cuda()

    def forward(self):
        return DotFunction.apply(self.input_left, self.input_right)
        # return self.input_left.dot(self.input_right)
        # return (
        #     DotFunction.apply(self.input_left, self.input_right),
        #     self.input_left.dot(self.input_right),
        # )

    # def forward(self):
    #     return self.input_left + self.input_right

module = Dot(128).cuda()

print(module())

module().sum().backward()

print(module.input_right.grad)
