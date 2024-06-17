import numpy as np
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor

def parallel_interpolation(xs, x, features, interp_method):
    '''
    xs : (N \times M) array # N represents the number of samples and must coincide with the N in features.
    x : dim long 1D array or (dim, 1) array.
    features : (N \times dim) array # each row represents the values at "x".
    '''
    print("Begin to do the interpolation")
    if interp_method == "linear":
        return np.vstack([np.interp(_x, np.ravel(x), _y) for _x, _y in zip(xs, features)])

    with ProcessPoolExecutor() as executor:
        res = list(executor.map(
            lambda _x, _y: interpolate.interp1d(
                np.ravel(x), _y, kind=interp_method, copy=False, assume_sorted=True
            )(_x),
            zip(xs, features)
        ))
    res = np.vstack(res)
    print("Finish the interpolation.")
    return res