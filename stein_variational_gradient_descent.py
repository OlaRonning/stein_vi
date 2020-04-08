import torch
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns


class SteinVariationalGradientDescent:
    def __init__(self, num_steps, step_size, target_dist, fudge_factor=1e-8, kernel='rbf', viz=True, alpha=0.9):
        self.num_steps = num_steps
        self.step_size = step_size
        self.fugdge_factor = fudge_factor
        self.step_size = step_size
        self.viz = viz
        self.alpha = alpha
        if kernel == 'rbf':
            self.kernel = self._rbf_kernel
        else:
            # TODO: Matrix kernel
            raise NotImplemented
        self.target_dist = target_dist

    def execute(self, particles):
        # add momentum term
        n = particles.shape[0]  # [n,..]

        gradient_history = torch.zeros((n, 1), dtype=torch.float)
        for step in range(self.num_steps):
            grad = self._functional_gradient(particles, n)
            gradient_history *= self.alpha
            gradient_history += ((not bool(step)) + bool(step) * (1 - self.alpha)) * grad ** 2
            rescale = 1 / torch.sqrt(gradient_history + self.fugdge_factor)
            particles += self.step_size * rescale * grad
            if self.viz and step % 10 == 0:
                plt.clf()
                sns.kdeplot(particles.squeeze().numpy())
                # plt.hist(target, bins=50, density=True)
                plt.show(block=False)
                plt.pause(0.01)

        return particles

    def _functional_gradient(self, particles, n):  # phi^{\star}
        kernel = self.kernel(particles).sum(1).unsqueeze(1)  # [n]
        grad_kernel = self._grad(self.kernel)(particles)  # [n]
        grad_logprob = self._grad(torch.log, self.target_dist)(particles)  # [n]
        attractive = kernel * grad_logprob
        repulsive = grad_kernel
        phi = attractive + repulsive
        phi[phi != phi] = 0
        result = phi/n  # emperical average
        return result

    @staticmethod
    def _grad(*args):
        def anno(x):
            # make leaf variables
            x_leaf = x.detach().requires_grad_(True)
            out = reduce(lambda o, f: f(o), reversed(args), x_leaf).squeeze()
            out.backward(torch.ones(out.shape).squeeze())
            return x_leaf.grad

        return anno

    @staticmethod
    def _rbf_kernel(x):
        n = x.shape[0]
        pw_dists = torch.pdist(x) ** 2  # Upper triangle
        med = torch.median(pw_dists)
        return SteinVariationalGradientDescent._upper_tria_to_full(n, torch.exp(
            -(torch.log(torch.tensor(n, dtype=float)) / (med ** 2 + 1e-8)) * pw_dists))

    @staticmethod
    def _upper_tria_to_full(n, triag_vec):
        matrix = torch.zeros((n, n), dtype=triag_vec.dtype)
        indices = torch.triu_indices(n, n, 1)
        matrix[indices[0], indices[1]] = triag_vec
        return matrix + matrix.T
