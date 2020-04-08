""" Bayesian Inference via Variational Gradient Descent """
from torch.distributions.normal import Normal
import math
from viz import plot_system
from stein_variational_gradient_descent import SteinVariationalGradientDescent
from utils import *


def gaussian_pdf(mu, sigma):
    def pdf(x):
        return (sigma * torch.sqrt(2 * torch.tensor([math.pi]))) ** -1 * torch.exp(
            -.5 * ((x - mu) / sigma) ** 2)

    return pdf


if __name__ == '__main__':
    # Initial toy target distribution
    pdf_1 = gaussian_pdf(to_tensor([0.0]), to_tensor([1.0]))
    pdf_2 = gaussian_pdf(to_tensor([3.0]), to_tensor([0.5]))
    target_dist = lambda x: 1 / 3 * pdf_1(x) + 2 / 3 * pdf_2(x)

    ### INPUT ###

    initial_dist = Normal(to_tensor([-10.]), to_tensor([1.]))  # could be arbitrary (not all zeros)
    num_particles = 100  # number of particles
    particles = initial_dist.sample((num_particles,)).squeeze().unsqueeze(1)

    ### ALGORITHM ###

    plot_system(target_dist, particles)
    sgb = SteinVariationalGradientDescent(num_steps=2000, step_size=0.5, kernel='rbf', target_dist=target_dist)
    particles = sgb.execute(particles)

    plot_system(target_dist, particles)
