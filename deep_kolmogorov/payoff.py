import abc
import numpy as np
from sklearn import gaussian_process as gp
from scipy.interpolate import interpn
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def _interp(args):
    points, _res, _x = args
    return interpn(points, _res, _x, bounds_error=False, fill_value=None)

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

    def __init__(self, sensor, grids=None, kernel="RBF", length_scale=10, var_scale=1):
        self.x = sensor
        self.points = None
        if grids is not None:
            self.dim = sensor.shape[-1]
            self.len = grids.shape[0]
            self.points = [grids.ravel() for _ in range(self.dim)]
            self.grids = torch.from_numpy(np.column_stack([c.ravel() for c in np.meshgrid(*self.points, indexing="ij")]))
        else:
            self.grids = self.x
        self.N = self.grids.shape[0]
        if kernel == "RBF":
            K = var_scale * gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = var_scale * gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.grids)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        u = np.random.randn(self.N, size)
        res = np.dot(self.L, u).T
        return res

    # def interp(self, xs, payoff):
    #     if payoff.device.type != "cpu":
    #         payoff = np.array(payoff.cpu(), dtype=str(payoff.cpu().dtype).split(".")[-1])
    #     res = payoff.reshape(payoff.shape[0], *[self.len for _ in range(self.dim)])
    #     if xs is not None:
    #         if xs.device.type != "cpu":
    #             xs = np.array(xs.cpu(), dtype=str(xs.cpu().dtype).split(".")[-1])
    #         sample = [interpn(self.points, _res, _x, bounds_error=False, fill_value=None) for _x,_res in zip(xs, res)]
    #         # with ThreadPoolExecutor() as executor:
    #         #     try:
    #         #         sample = list(executor.map(_interp, zip([self.points]*len(xs), res, xs)))
    #         #     except Exception as e:
    #         #         print(f"Task generated an exception {e}.")
    #     else:
    #         x = np.array(self.x.cpu(), dtype=str(self.x.cpu().dtype).split(".")[-1])
    #         sample = [interpn(self.points, _res, x) for _res in res]
    #         # with ThreadPoolExecutor() as executor:
    #         #     try:
    #         #         sample = list(executor.map(_interp, zip([self.points]*len(res), res, [x]*len(res))))
    #         #     except Exception as e:
    #         #         print(f"Task generated an exception {e}.")
    #     return np.vstack(sample)

    def interp(self, xs, payoff):
        # convert point to 1d then interpolate
        if payoff.device.type != "cpu":
            payoff = np.array(payoff.cpu(), dtype=str(payoff.cpu().dtype).split(".")[-1])
        res = payoff.reshape(payoff.shape[0], *[self.len for _ in range(self.dim)])

        sorted_index = np.argsort(torch.exp(torch.mean(torch.log(self.grids), dim=-1)).cpu())
        grids_sorted = torch.exp(torch.mean(torch.log(self.grids), dim=-1)).cpu()[sorted_index]
        if xs is not None:
            if xs.device.type != "cpu":
                xs = np.array(xs.cpu(), dtype=str(xs.cpu().dtype).split(".")[-1])
            sample = [np.interp(np.exp(np.mean(np.log(_x),axis=-1)), grids_sorted, _res.ravel()[sorted_index]) for _x,_res in zip(xs, res)]
            # with ThreadPoolExecutor() as executor:
            #     try:
            #         sample = list(executor.map(_interp, zip([self.points]*len(xs), res, xs)))
            #     except Exception as e:
            #         print(f"Task generated an exception {e}.")
        else:
            x = np.array(self.x.cpu(), dtype=str(self.x.cpu().dtype).split(".")[-1])
            sample = [np.interp(np.exp(np.mean(np.log(x),axis=-1)), grids_sorted, _res.ravel()[sorted_index]) for _res in res]
            # with ThreadPoolExecutor() as executor:
            #     try:
            #         sample = list(executor.map(_interp, zip([self.points]*len(res), res, [x]*len(res))))
            #     except Exception as e:
            #         print(f"Task generated an exception {e}.")
        return np.vstack(sample)
            



PAYOFFS = {pf.__name__: pf for pf in FunctionSpace.get_subclasses()}