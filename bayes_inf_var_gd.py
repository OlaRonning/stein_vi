""" Bayesian Inference via Variational Gradient Descent """
from torch.distributions.normal import Normal
import torch
from functools import reduce
import math
import numpy as np
import matplotlib.pyplot as plt

to_tensor = torch.tensor


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


class BayesInference:
    def __init__(self, num_steps, step_size, kernel, target_dist):
        self.num_steps = num_steps
        if type(step_size) == float:  # check propertype
            # make constant function 
            eta = step_size
            step_size = lambda _: eta
        self.step_size = step_size
        self.kernel = kernel
        self.target_dist = target_dist

    def variational_gd(self, particles):
        n = particles.shape[0]  # [n,..]

        for step in range(self.num_steps):
            grad = self.step_size(step) * self._functional_gradient(particles, n)
            particles += grad

        return particles

    def _functional_gradient(self, particles, n):  # phi^{\star}
        grad_kernel = self._grad(self.kernel)(particles)  # [n,n]
        kernel = self.kernel(particles)  # [n,n]
        grad_logprob = self._grad(torch.log, self.target_dist)(particles).expand((n, n))  # [n,n]
        phi = kernel * grad_logprob + grad_kernel
        result = 1 / n * torch.sum(phi, dim=1)  # emperical average
        return result.unsqueeze(1)

    @staticmethod
    def _grad(*args):
        def anno(x):
            # make leaf variables
            x_leaf = x.detach().requires_grad_(True)
            out = reduce(lambda o, f: f(o), reversed(args), x_leaf).squeeze()
            out.backward(torch.ones(out.shape).squeeze())
            return x_leaf.grad

        return anno


def upper_tria_to_full(n, triag_vec):
    # TODO: make smarter
    matrix = torch.zeros((n, n), dtype=triag_vec.dtype)
    indices = torch.triu_indices(n, n, 1)
    matrix[indices[0], indices[1]] = triag_vec
    return matrix + matrix.T


def plot_system(target, particles):
    plt.scatter(particles, np.arange(particles.shape[0]))
    plt.show()


def normal_pdf(mu, sigma):
    def anno(x):
        return (sigma * torch.sqrt(2 * torch.tensor([math.pi]))) ** -1 * torch.exp(
            -.5 * ((x - mu) / sigma) ** 2)  # TODO: check this calculation

    return anno


def rbf_kernel(x):
    n = x.shape[0]
    pw_dists = torch.pdist(x) ** 2  # Upper triangle
    med = torch.median(pw_dists)
    return upper_tria_to_full(n, torch.exp(
        -torch.sqrt(torch.tensor(n, dtype=float)) / med ** 2 * pw_dists))


if __name__ == '__main__':
    ### INPUT ###
    f1 = normal_pdf(to_tensor([0.0]), to_tensor([1.0]))
    f2 = normal_pdf(to_tensor([3.0]), to_tensor([0.5]))
    target_dist = lambda x: 1 / 3 * f1(x) + 2 / 3 * f2(x)

    initial_dist = Normal(to_tensor([-10.]), to_tensor([1.]))  # could be arbitary
    num_particles = 20  # number of particles

    particles = initial_dist.sample((num_particles,))

    ### ALGORITHM ###

    plot_system(target_dist, particles)
    bi = BayesInference(num_steps=1000, step_size=0.001, kernel=rbf_kernel, target_dist=target_dist)
    particles = bi.variational_gd(particles)

    plot_system(target_dist, particles)
