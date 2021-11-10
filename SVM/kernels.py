import numpy as np


# The best way to implement this would be with an abstract base class to enforce
# that all Kernels require a calc_kernel method

class LinearKernel:

    def calc_kernel(self, X: np.ndarray):
        return np.dot(X, X.T) # Gives a matrix of dot products between all x_i's


class GaussianKernel:

    def __init__(self, gamma: float):
        self.gamma = gamma # A hyperparamter determining Gaussian scaling

    def calc_kernel(self, X: np.ndarray):
        # k(x_i, x_j) = exp(-||x_i - x_j||^2 / gamma)
        x_i_j_diff = self.create_rolled_x_diff_array(X)
        return np.exp(-np.square(np.linalg.norm(x_i_j_diff)) / self.gamma)

    def create_rolled_x_diff_array(self, X: np.ndarray):
        # creates a matrix over which values can be summed to minimize for loops
        x_i_j_diff = np.array([])
        for j in range(len(X)):
            x_i_j_diff = np.append(x_i_j_diff, X - np.roll(X, j, axis=0))
        return x_i_j_diff
