import abc
import numpy as np
from sklearn import gaussian_process as gp


class FunctionSpace(abc.ABC):

    @abc.abstractmethod
    def random(self, size):
        """Generate feature vectors of random functions.

        Args:
            size (int): The number of random functions to generate.

        Returns:
            A NumPy array of shape (`size`, n_features).
        """
    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    # @abc.abstractmethod
    # def eval_one(self, feature, x):
    #     """Evaluate the function at one point.

    #     Args:
    #         feature: The feature vector of the function to be evaluated.
    #         x: The point to be evaluated.

    #     Returns:
    #         float: The function value at `x`.
    #     """

    # @abc.abstractmethod
    # def eval_batch(self, features, xs):
    #     """Evaluate a list of functions at a list of points.

    #     Args:
    #         features: A NumPy array of shape (n_functions, n_features). A list of the
    #             feature vectors of the functions to be evaluated.
    #         xs: A NumPy array of shape (n_points, dim). A list of points to be
    #             evaluated.

    #     Returns:
    #         A NumPy array of shape (n_functions, n_points). The values of
    #         different functions at different points.
    #     """



class GRF(FunctionSpace):
    """Gaussian random field (Gaussian process).

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    Args:
        T (float): `T` > 0. The domain is [0, `T`].
        kernel (str): Name of the kernel function. "RBF" (radial-basis function kernel,
            squared-exponential kernel, Gaussian kernel), "AE"
            (absolute exponential kernel), or "ExpSineSquared" (Exp-Sine-Squared kernel,
            periodic kernel).
        length_scale (float): The length scale of the kernel.
        N (int): The size of the covariance matrix.
        interp (str): The interpolation to interpolate the random function. "linear",
            "quadratic", or "cubic".
    """

    def __init__(self, sensor, kernel="RBF", length_scale=10, var_scale=1):
        self.x = sensor
        self.N = self.x.shape[0]
        if kernel == "RBF":
            K = var_scale * gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = var_scale * gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        u = np.random.randn(self.N, size)
        return np.dot(self.L, u).T
    



PAYOFFS = {pf.__name__: pf for pf in FunctionSpace.get_subclasses()}