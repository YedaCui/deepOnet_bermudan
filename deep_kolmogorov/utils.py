import numpy as np
import torch
from scipy import interpolate
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.stats.qmc import Sobol


def _interp(args):
    x, _x, _y = args
    return interpolate.LinearNDInterpolator(x, _y)(_x)

# def _interp(args):
#     x, _x, _y = args
#     return interpolate.Rbf(*x.T, _y, function='multiquadric')(*_x)


def parallel_interpolation(xs, x, features, interp_method):
    '''
    xs : (N \times M) array # N represents the number of samples and must coincide with the N in features.
    x : dim long 1D array or (dim, 1) array.
    features : (N \times dim) array # each row represents the values at "x".
    '''
    def move_to_cpu_if_tensor(var):
        if torch.is_tensor(var):
            return var.cpu()
        else:
            return var
    xs_device = xs.device
    print(f"The device of xs is {xs_device}.")
    # move all the thing to cpu
    xs = move_to_cpu_if_tensor(xs)
    x = move_to_cpu_if_tensor(x)
    features = move_to_cpu_if_tensor(features)


    
    print("Begin to do the interpolation")
    if xs.shape[1] == 1:
        print("the sensor shape [1] is 1")
        if interp_method == "linear":
            res = np.vstack([np.interp(_x, np.ravel(x), _y) for _x, _y in zip(xs, features)])
    else:
        if interp_method == "linear":
            # interps = [interpolate.LinearNDInterpolator(x, _y) for _y in features]
            # res = [_interp(_x) for _interp, _x in zip(interps, xs)]

            # interps = [interpolate.Rbf(*x.T, _y, function='multiquadric') for _y in features]
            # res = [_interp(*_x) for _interp, _x in zip(interps, xs)]

            with ProcessPoolExecutor() as executor:
                try:
                    res = list(executor.map(_interp, zip([x]*len(xs), xs,features)))
                except Exception as e:
                    print(f"Task generated an exception {e}.")
            res = np.vstack(res)
    print("Finish the interpolation.")
    res = torch.from_numpy(res).to(xs_device)
    return res

def CN_bermudan_1D(cpflag, K, T, num_ex, vol, r, d, N = 2000, x_max=3, S0=10):
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
    device = K.device
    # grid along x dimension:
    X = torch.linspace(-x_max,x_max,N+1,device=device)
    #number of steps along x
    dx = 2*x_max/N

    mu = (r-d-0.5*vol*vol)
    #number of time steps
    J = int((int(1000/num_ex) + 1) * num_ex)
    dt = T/J
    dJ = int(J/num_ex)
    
    # set up the matrix
    a = 0.25*dt*vol*vol/(dx*dx)
    b = 0.25*dt*mu/dx
    c = 0.5*dt*r
    A, B = torch.zeros((a.shape[0],N+1,N+1),device=device), torch.zeros((a.shape[0],N+1,N+1),device=device)
    for _n in range(a.shape[0]):
        _a, _b, _c = a[_n], b[_n], c[_n]
        A[_n,:,:] = (1+_c+2*_a)*torch.eye(N+1,device=device) + (-_a-_b)*torch.diag(torch.ones(N,device=device),1) + (_b-_a)*torch.diag(torch.ones(N,device=device),-1)
        B[_n,:,:] = (1-_c-2*_a)*torch.eye(N+1,device=device) + (_a+_b)*torch.diag(torch.ones(N,device=device),1) + (_a-_b)*torch.diag(torch.ones(N,device=device),-1)
    Ainv = torch.linalg.inv(A)
    
    if cpflag == 'call':
        # Option payoff at maturity
        V = torch.clamp(S0*torch.exp(X).reshape(1,-1) - K.reshape(-1,1),0,1e10).unsqueeze(1).permute(0,2,1)
    elif cpflag == 'put':
        V = torch.clamp(K.reshape(-1,1) - S0*torch.exp(X).reshape(1,-1),0,1e10).unsqueeze(1).permute(0,2,1)
    
    V0 = V.clone()
    Vs = []
    for j in range(1, J+1):
        V = B @ V
        V = Ainv @ V
        # apply early exercise boundary conditions:
        if (j%dJ==0) and j!=J:
            Vs.append(V.clone().squeeze(-1))
            V = torch.where(V>V0,V,V0)
    Vs.append(V.clone().squeeze(-1))
    return S0*torch.exp(X), Vs[::-1]

def CN_bermudan_ND(cpflag, K, T, num_ex, vol, r, d, rho, N = 2000, x_max=3, S0=10):
    '''
    This only works for Geometirc Basket mean bermudan options.
    '''

    device = K.device
    # grid along x dimension:
    X = torch.linspace(-x_max,x_max,N+1,device=device)
    #number of steps along x
    dx = 2*x_max/N

    d_bar = torch.mean(d, dim=-1)
    n = vol.shape[-1] # the dimension of S_t
    batch_size = vol.shape[0]
    sig_bar = torch.mean(vol**2, dim=1)
    RHO = rho.view(batch_size, 1, 1).expand(batch_size, n, n).clone()
    RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
    sig_tilde = 1/n**2 * torch.sum(vol.unsqueeze(2) * RHO * vol.unsqueeze(1), (1,2)).flatten()
    mu = (r-d_bar-0.5*sig_bar)
    #number of time steps
    J = int((int(1000/num_ex) + 1) * num_ex)
    dt = T/J
    dJ = int(J/num_ex)
    
    # set up the matrix
    a = 0.25*dt*sig_tilde/(dx*dx)
    b = 0.25*dt*mu/dx
    c = 0.5*dt*r
    A, B = torch.zeros((a.shape[0],N+1,N+1),device=device), torch.zeros((a.shape[0],N+1,N+1),device=device)
    for _n in range(a.shape[0]):
        _a, _b, _c = a[_n], b[_n], c[_n]
        A[_n,:,:] = (1+_c+2*_a)*torch.eye(N+1,device=device) + (-_a-_b)*torch.diag(torch.ones(N,device=device),1) + (_b-_a)*torch.diag(torch.ones(N,device=device),-1)
        B[_n,:,:] = (1-_c-2*_a)*torch.eye(N+1,device=device) + (_a+_b)*torch.diag(torch.ones(N,device=device),1) + (_a-_b)*torch.diag(torch.ones(N,device=device),-1)
    Ainv = torch.linalg.inv(A)
    
    if cpflag == 'call':
        # Option payoff at maturity
        V = torch.clamp(S0*torch.exp(X).reshape(1,-1) - K.reshape(-1,1),0,1e10).unsqueeze(1).permute(0,2,1)
    elif cpflag == 'put':
        V = torch.clamp(K.reshape(-1,1) - S0*torch.exp(X).reshape(1,-1),0,1e10).unsqueeze(1).permute(0,2,1)
    
    V0 = V.clone()
    Vs = []
    for j in range(1, J+1):
        V = B @ V
        V = Ainv @ V
        # apply early exercise boundary conditions:
        if (j%dJ==0) and j!=J:
            Vs.append(V.clone().squeeze(-1))
            V = torch.where(V>V0,V,V0)
    Vs.append(V.clone().squeeze(-1))
    return S0*torch.exp(X), Vs[::-1]

def MP_grid(a=0.01, s=10, b=80, g1=10, g2=5, n=50):
    n = n//2
    c1 = np.arcsinh((a - s) / g1)
    c2 = np.arcsinh((b - s) / g2)
    
    linspace1 = np.linspace(1, n, num=n)
    linspace2 = linspace1 / n  # create a linear space from 1/n to 1
    linspace3 = (n - linspace1) / (n - 1)  # create a linear space from 0 to 1, adjusted for indexing from 1
    
    Gblock1 = s + g1 * np.sinh(c1 * linspace3)
    Gblock2 = s + g2 * np.sinh(c2 * linspace2)
    
    Gblock = np.concatenate([Gblock1, Gblock2],axis=0)
    return Gblock.reshape(-1,1).astype(np.float32)
    
def qmc_grid(a=0.01, b=80, d=3, n=50, shiftbymp=True, s=10, g1=10, g2=5, seed=0):
    # if shiftbymp is True, shift the samples by MP nonlinear mapping.
    

    sampler = Sobol(d, seed=seed)
    samples = sampler.random(n)

    if not shiftbymp:
        res = a + (b-a)*samples
    else:
        c1 = np.arcsinh((a - s) / g1)
        c2 = np.arcsinh((b - s) / g2)
        res = np.where(samples <= 0.5, s + g1 * np.sinh(c1 * (0.5-samples)*2), s + g2 * np.sinh(c2 * (samples-0.5)*2))

    sorted_indices = np.lexsort([res[:, i] for i in range(d-1, -1, -1)])
    res = res[sorted_indices]

    return res.astype(np.float32)
        

        
        