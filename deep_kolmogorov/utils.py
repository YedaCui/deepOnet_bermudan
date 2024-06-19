import numpy as np
import torch
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

def CN_bermudan_1D(cpflag, K, T, num_ex, vol, r, d, N = 1000, x_max=3, S0=10):
    '''
    args:
    cpflag: str, "call" or "put".
    K: 1D array
    T: float
    num_exe: number of exercise dates.
    vol: 1D array
    r: 1D array
    d: 1D array, the dividend paying rate.

    Attension: the ND arrays should have the same shape.
    '''
    # grid along x dimension:
    X = torch.linspace(-x_max,x_max,N+1)
    #number of steps along x
    dx = 2*x_max/N

    mu = (r-d-0.5*vol*vol)

    #number of time steps
    J = 1000
    dt = T/J
    dJ = int(J/num_ex)
    
    # set up the matrix
    a = 0.25*dt*vol*vol/(dx*dx)
    b = 0.25*dt*mu/dx
    c = 0.5*dt*r
    A, B = torch.zeros((a.shape[0],N+1,N+1)), torch.zeros((a.shape[0],N+1,N+1))
    for _n in range(a.shape[0]):
        _a, _b, _c = a[_n], b[_n], c[_n]
        A[_n,:,:] = (1+_c+2*_a)*torch.eye(N+1) + (-_a-_b)*torch.eye(N+1,k=1) + (_b-_a)*torch.eye(N+1,k=-1)
        B[_n,:,:] = (1-_c-2*_a)*torch.eye(N+1) + (_a+_b)*torch.eye(N+1,k=1) + (_a-_b)*torch.eye(N+1,k=-1)
    Ainv = torch.linalg.inv(A)
    
    if cpflag == 'call':
        # Option payoff at maturity
        V = torch.clamp(S0*torch.exp(X).reshape(1,-1) - K.reshape(-1,1),0,1e10).unsqueeze(1).transpose(0,2,1)
    elif cpflag == 'put':
        V = torch.clamp(K.reshape(-1,1) - S0*torch.exp(X).reshape(1,-1),0,1e10).unsqueeze(1).transpose(0,2,1)
    
    V0 = V.clone()
    for j in range(1, J+1):
        V = B @ V
        V = Ainv @ V
        # apply early exercise boundary conditions:
        if (j%dJ==0) and j!=J:
            V = torch.where(V>V0,V,V0)
    return S0*torch.exp(X), V.squeeze(-1)
