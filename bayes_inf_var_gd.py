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
    pdf_1 = gaussian_pdf(to_tensor(2.0), to_tensor(1.))
    pdf_2 = gaussian_pdf(to_tensor(-2.0), to_tensor(1.))
    target_dist = lambda x: 1 / 4 * pdf_1(x) + 3 / 4 * pdf_2(x)

    ### INPUT ###

    initial_dist = Normal(to_tensor([-10.]), to_tensor([1.]))  # could be arbitrary (not all zeros)
    num_particles = 99  # number of particles
    particles = initial_dist.sample((num_particles,)).squeeze().unsqueeze(1)
    # particles = torch.ones((num_particles,1), dtype=torch.float)

    print(pdf_1(particles))
    ### ALGORITHM ###

    # plot_system(particles)
    sgb = SteinVariationalGradientDescent(num_steps=500, step_size=1., kernel='rbf', target_dist=target_dist)
    particles = sgb.execute(particles)
    print(particles)
    plot_system(particles)
