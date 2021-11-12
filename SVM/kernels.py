import numpy as np


# The best way to implement this would be with an abstract base class to enforce
# that all Kernels require a calc_kernel method

class LinearKernel:

    def calc_kernel(self, X1: np.ndarray, X2: np.ndarray):
        # X1, X2 = _pad_smaller_matrix_with_0s(X1, X2)
        return np.dot(X1, X2.T)  # Gives a matrix of dot products between all x_i's


class GaussianKernel:

    def __init__(self, gamma: float):
        self.gamma = gamma  # A hyperparamter determining Gaussian scaling

    def calc_kernel(self, X1: np.ndarray, X2: np.ndarray):
        # k(x_i, x_j) = exp(-||x_i - x_j||^2 / gamma)
        # X1, X2 = _pad_smaller_matrix_with_0s(X1, X2)
        x_i_j_diff = self.create_rolled_x_diff_array(X1, X2)
        # print(np.linalg.norm(x_i_j_diff, axis=1).shape)
        print('KERNEL SHAPE', np.exp(-np.square(np.linalg.norm(x_i_j_diff, axis=1)) / self.gamma).shape)
        return np.exp(-np.square(np.linalg.norm(x_i_j_diff, axis=1)) / self.gamma)

    def create_rolled_x_diff_array(self, X1: np.ndarray, X2: np.ndarray):
        # creates a matrix over which values can be summed to minimize for loops
        x_i_j_diff = np.empty((0, len(X1[0]), len(X1)), float).T
        for j in range(len(X1)):
            x_i_j_diff = np.dstack([x_i_j_diff, X1 - np.roll(X2, j, axis=0)])
        # print(j)
        # if len(X1) <= len(X2):
        #     for j in range(len(X1)):
        #         # print(np.tile(X1[j], (len(X2), 1)))
        #         print((np.tile(X1[j], (len(X2), 1)) - X2).shape)
        #         print((X1 - np.roll(X2, j, axis=0)).shape)
        #         x_i_j_diff = np.append(x_i_j_diff, np.tile(X1[j], (len(X2), 1)) - X2)
        # else:
        #     for j in range(len(X2)):
        #         x_i_j_diff = np.append(x_i_j_diff, X1 - np.tile(X2[j], (len(X1), 1)))
        # print('TEST SHAPE', x_i_j_diff.shape)
        # print(x_i_j_diff.shape)
        return x_i_j_diff


# def _pad_smaller_matrix_with_0s(X1: np.ndarray, X2: np.ndarray) -> (np.ndarray, np.ndarray):
#     row_diff = len(X1) - len(X2)
#     if row_diff == 0:
#         return X1, X2
#     elif row_diff < 0:
#         X1 = np.pad(X1, pad_width=((0, -row_diff),(0, 0)))
#     elif row_diff > 0:
#         X2 = np.pad(X2, pad_width=((0, row_diff),(0, 0)))
#     return X1, X2


# class Kernel_ARD:
#     def __init__(self, jitter=0):
#         self.jitter = tf.constant(jitter, dtype=tf_type)
    
#     def matrix(self, X, amp, ls):
#         K = self.cross(X,X,amp, ls)
#         K = K + self.jitter * tf.eye(tf.shape(X)[0], dtype=tf_type)
#         return K

#     def cross(self, X1, X2, amp, ls):
#         norm1 = tf.reshape(tf.reduce_sum(X1**2, 1), [-1, 1])
#         norm2 = tf.reshape(tf.reduce_sum(X2**2, 1), [1, -1])
#         K = norm1 - 2.0 * tf.matmul(X1, tf.transpose(X2)) + norm2
#         K = amp * tf.exp(-1.0 * K / ls)
#         return K
