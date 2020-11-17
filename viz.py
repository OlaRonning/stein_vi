import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from stein_variational_gradient_descent import SteinVariationalGradientDescent

image_dir = 'images'


def plot_system(target_dist, particles, iter=0):
    lin_space = torch.linspace(min(-10, min(particles.squeeze().numpy())), max(10, min(particles.squeeze().numpy())),
                               500)

    plt.clf()
    sns.kdeplot(particles.squeeze().numpy())
    plt.plot(lin_space, target_dist(lin_space))
    plt.title(f'Iteration {iter}')
    plt.savefig(f'{image_dir}/system_at_{iter}.png', format='png')
    plt.show(block=False)
    plt.pause(0.1)

