import abc
import numpy as np
from sklearn import gaussian_process as gp
from scipy import linalg, interpolate


class FunctionSpace(abc.ABC):
    """Function space base class.

    Example:

        .. code-block:: python

            space = dde.data.GRF()
            feats = space.random(10)
            xs = np.linspace(0, 1, num=100)[:, None]
            y = space.eval_batch(feats, xs)
    """

    @abc.abstractmethod
    def random(self, size, mf):
        """Generate feature vectors of random functions.

        Args:
            size (int): The number of random functions to generate.
            mf (function): the mean function.

        Returns:
            A NumPy array of shape (`size`, n_features).
        """

    @abc.abstractmethod
    def eval_one(self, feature, x):
        """Evaluate the function at one point.

        Args:
            feature: The feature vector of the function to be evaluated.
            x: The point to be evaluated.

        Returns:
            float: The function value at `x`.
        """

    @abc.abstractmethod
    def eval_batch(self, features, xs):
        """Evaluate a list of functions at a list of points.

        Args:
            features: A NumPy array of shape (n_functions, n_features). A list of the
                feature vectors of the functions to be evaluated.
            xs: A NumPy array of shape (n_points, dim). A list of points to be
                evaluated.

        Returns:
            A NumPy array of shape (n_functions, n_points). The values of
            different functions at different points.
        """



class GRF(FunctionSpace):
    """Gaussian random field (Gaussian process) in 1D.

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

    def __init__(self, e1=0.01, e2=100, kernel="RBF", length_scale=10, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(e1, e2, num=N)[:, None] # shape (N,1)
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size, mf=lambda x: np.zeros_like(x)):
        u = np.random.randn(self.N, size)
        return np.dot(self.L, u).T + mf(self.x.T)

    def eval_one(self, feature, x):
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), feature)
        f = interpolate.interp1d(
            np.ravel(self.x), feature, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_batch(self, features, xs):
        if self.interp == "linear":
            return np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(xs).T,
            features,
        )
        return np.vstack(list(res))