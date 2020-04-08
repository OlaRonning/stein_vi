import torch
from functools import reduce


class SteinVariationalGradientDescent:
    def __init__(self, num_steps, step_size, target_dist, fudge_factor=1e-8, kernel='rbf'):
        self.num_steps = num_steps
        if type(step_size) == float:  # check propertype
            # make constant function
            eta = step_size
            step_size = lambda _: eta
        self.fugdge_factor = fudge_factor
        self.step_size = step_size
        if kernel == 'rbf':
            self.kernel = self._rbf_kernel
        else:
            # TODO: Matrix kernel
            raise NotImplemented
        self.target_dist = target_dist

    def execute(self, particles):
        # add momentum term
        n = particles.shape[0]  # [n,..]

        gradient_history = torch.zeros((n,1), dtype=torch.float)
        for step in range(self.num_steps):
            grad = self._functional_gradient(particles, n)
            gradient_history += grad ** 2
            rescale = 1 / torch.sqrt(gradient_history + self.fugdge_factor)
            particles += self.step_size(step) * rescale * grad

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

    @staticmethod
    def _rbf_kernel(x):
        n = x.shape[0]
        pw_dists = torch.pdist(x) ** 2  # Upper triangle
        med = torch.median(pw_dists)
        return SteinVariationalGradientDescent._upper_tria_to_full(n, torch.exp(
            -torch.sqrt(torch.tensor(n, dtype=float)) / med ** 2 * pw_dists))

    @staticmethod
    def _upper_tria_to_full(n, triag_vec):
        # TODO: make smarter
        matrix = torch.zeros((n, n), dtype=triag_vec.dtype)
        indices = torch.triu_indices(n, n, 1)
        matrix[indices[0], indices[1]] = triag_vec
        return matrix + matrix.T
