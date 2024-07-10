# import necessary libraries
from abc import ABC, abstractmethod
import torch
import itertools


class RegMethod(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def pred(self, x_new):
        pass

class PolynomialReg(RegMethod):
    def __init__(self, degree, dimension):
        super().__init__()
        self.degree = degree
        self.dimension = dimension
        self.terms = self.generate_degree_combinations(dimension,degree)
    
    @staticmethod
    def generate_degree_combinations(d, g):
        """
        Generate all combinations of polynomial degrees for `d` variables up to degree `g`.
        
        Args:
        d (int): Number of variables.
        g (int): Maximum degree of the polynomial.
        
        Returns:
        List of tuples representing the powers of each variable in the term.
        """
        terms = [tuple(0 for _ in range(d))]
        # terms = []
        for total_degree in range(1, g + 1):
            for powers in itertools.combinations_with_replacement(range(total_degree + 1), d):
                if sum(powers) == total_degree:
                    terms.extend(set(itertools.permutations(powers)))
        return terms
    
    @staticmethod
    def polynomial_features(input_tensor, degree_combinations):
        """
        Generate polynomial features for input tensor.
        
        Args:
        input_tensor (Tensor): Input tensor of shape (num_samples, num_features)
        degree_combinations (list): combonation of degrees features.
        
        Returns:
        Tensor: Extended feature matrix including polynomial terms.
        """
        poly_features = [torch.prod(input_tensor**torch.tensor(powers, dtype=input_tensor.dtype, device=input_tensor.device), dim=-1, keepdim=True) for powers in degree_combinations]
        return torch.concat(poly_features, dim=-1)

    def get_input(self, x):
        """
        x : 2D tensor
        """
        return self.polynomial_features(x, self.terms)
    
    def fit(self, x, y):
        """
        x: list of 2D tensor
        y: list of 1D tensor
        """
        x = [self.get_input(_x) for _x in x]
        self.betas = [torch.pinverse(_x.T @ _x) @ _x.T @ _y.reshape(-1,1)  for _x,_y in zip(x,y)]
        
    def pred(self, x_new: list[torch.tensor]) -> list[torch.tensor]:
        """
        x_new 
        """
        x_new = [self.get_input(_x) for _x in x_new]
        return [(_x @ _beta).squeeze(-1) for _beta,_x in zip(self.betas,x_new)]


class LSMC:
    def __init__(self, pde, regmethod, T, opt_type, num_ex, num_sim):
        self.pde = pde
        self.regmethod = regmethod
        self.T = T
        self.opt_type = opt_type
        self.num_ex = num_ex
        self.dt = self.T/self.num_ex
        self.num_sim = num_sim

    def pricing(self, batch):
        batch_sim = {_param:batch[_param].repeat_interleave(self.num_sim, dim=0) for _param in batch.keys()}
        x_sim = [batch_sim["s"].clone()]
        batch_sim["t"].fill_(self.dt)
        for _ in range(self.num_ex):
            x_sim.append(self.pde.get_X(x_sim[-1], batch_sim["r"], batch_sim["q"], batch_sim["sigma"], batch_sim["t"], batch_sim["rho"]))
        x_sim = torch.stack(x_sim, dim=1)
        x_sim = x_sim.reshape(self.num_sim, batch["K"].shape[0], self.num_ex+1, x_sim.shape[-1]).permute(1,0,2,3)
        K = batch_sim["K"].reshape(self.num_sim, batch["K"].shape[0], -1).permute(1,0,2).unsqueeze(2)
        r = batch_sim["r"].reshape(self.num_sim, batch["K"].shape[0], -1).permute(1,0,2).squeeze(-1)

        payoff_sim = self.pde.get_payoff(x_sim, K, self.opt_type).squeeze(-1) # with dimension num_batch \times num_sim \times num_ex + 1

        Y = payoff_sim[...,-1]
        for i in range(self.num_ex-1, -1, -1):
            Y *= torch.exp(-r*self.dt)
            if i == 0:
                break
            
            holds = [_pf_sim > 0 for _pf_sim in payoff_sim[...,i]]
            x_sim_holds = [_x_sim[_hold, :] for _hold,_x_sim in zip(holds, x_sim[:,:,i,:])]
            Y_holds = [_Y[_hold] for _hold,_Y in zip(holds, Y)]

            self.regmethod.fit(x_sim_holds, Y_holds)
            contvalue = self.regmethod.pred(x_sim_holds)

            for j, _hold in enumerate(holds):
                Y[j, _hold] = torch.where(
                    payoff_sim[j, _hold, i] > contvalue[j], payoff_sim[j, _hold, i], Y[j, _hold]
                )
        return torch.mean(Y, dim=-1, keepdim=True), torch.std(Y, dim=-1, keepdim=True)/self.num_sim**0.5