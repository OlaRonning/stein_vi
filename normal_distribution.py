import torch


class Normal_Dist(torch.autograd.Function):
    """ NOT NEEDED BUT GOOD FOR REFERENCE"""

    @staticmethod
    def forward(context, x):
        result = 1 / torch.sqrt(2 * torch.tensor([math.pi])) * torch.exp(-1 / 2 * (x)) ** 2
        context.save_for_backward(result)
        return result

    @staticmethod
    def backward(context, grad_output):
        result, = context.saved_tensors
        return grad_output * result
