import torch
from torch.utils.data import Dataset, DataLoader
from .utils import *
import os
from abc import ABC, abstractmethod
import ray


class Data_Bermudan(Dataset):
    def __init__(self, pde, payoff, T, num_ex, option_type, batch_size, n_batches, frezed_params, interp_method, var_rescale=False, var_rescale_k=1, device="cpu"):
        self.pde = pde
        self.dimension = pde.hypercubes["s"].dims[0]
        self.payoff = payoff
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.frezed_params = frezed_params
        self.T = T
        self.num_ex = num_ex
        self.dt = self.T/self.num_ex
        self.option_type = option_type
        self.interp_method = interp_method
        self.var_rescale = var_rescale
        self.var_rescale_k = var_rescale_k
        self.device = device

    
    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        '''
        idx : useless arg in our case.
        '''
        
        dt_pde = next(iter(self.pde.dataloader(self.batch_size, 1, 'train', frezed_params=self.frezed_params))) # generate a batch pdes data of model parameters

        if torch.cuda.is_available():
            dt_pde = {_k: _v.to(self.device) for _k, _v in dt_pde.items()}

        device = dt_pde["s"].device
        data_batch = {_k: _v.repeat([self.num_ex] + [1] * (_v.dim()-1)) for _k,_v in dt_pde.items()}
        data_batch["t"] = torch.arange(0, self.T, self.dt, device=device).reshape(-1,1).repeat_interleave(dt_pde["s"].shape[0], dim=0)
        data_batch["x"] = self.pde.get_X(data_batch, data_batch["s"])
        data_batch["tau"] = self.T - data_batch["t"] - self.dt
        cont_value = self.pde.option_price(data_batch, self.payoff.x.reshape(self.dimension,-1), option_type=self.option_type)
        if self.var_rescale == True:
            var_rescale_const = self.var_rescale_k * (self.T - torch.arange(self.dt, self.T, self.dt)).reshape(-1,1).repeat_interleave(dt_pde["s"].shape[0],dim=0).to(cont_value.device)
            cont_value[:-dt_pde["s"].shape[0],:] = cont_value[:-dt_pde["s"].shape[0],:] + var_rescale_const * torch.from_numpy(self.payoff.random(cont_value[:-dt_pde["s"].shape[0],:].shape[0])).to(cont_value.device)
        else:
            cont_value += torch.from_numpy(self.payoff.random(cont_value.shape[0])).to(cont_value.device)
        cont_value[-dt_pde["s"].shape[0]:,:] = 0

        data_batch["payoff"] = torch.maximum(cont_value, self.pde.get_payoff(self.payoff.x.reshape(self.dimension,-1), data_batch["K"], opt_type=self.option_type))
        data_batch["t"].fill_(self.dt)
        xs = self.pde.get_X(data_batch, data_batch["x"])

        data_batch["y"] = torch.exp(- data_batch["r"] * self.dt) * parallel_interpolation(xs, self.payoff.x, data_batch["payoff"], interp_method=self.interp_method)

        return data_batch



    
    # def __getitem__(self, idx):
    #     '''
    #     idx : useless arg in our case.
    #     '''

    #     dt_pde = next(iter(self.pde.dataloader(self.batch_size, 1, 'train', frezed_params=self.frezed_params))) # generate a batch pdes data
    #     device = dt_pde["t"].device
    #     res = {"payoff": [],
    #            "y": []
    #            }
    #     for i in range(self.num_ex):
    #         dt_pde["t"].fill_(i*self.T/self.num_ex)
    #         dt_pde["x"] = self.pde.get_X(dt_pde)
    #         dt_pde["t"] += self.T/self.num_ex
    #         if i == self.num_ex-1:
    #             cont_value = torch.zeros(1,1, device=device)
    #         else:
    #             cont_value = self.pde.option_price(self.T - dt_pde["t"], self.payoff.x.reshape(1,-1), dt_pde["sigma"], dt_pde["r"], dt_pde["q"], dt_pde["K"], option_type=self.option_type)
    #             if self.var_rescale == True:
    #                 cont_value += self.var_rescale_k * (self.T - (i+1)*self.T/self.num_ex) * torch.from_numpy(self.payoff.random(cont_value.shape[0]))
    #             else:
    #                 cont_value += torch.from_numpy(self.payoff.random(cont_value.shape[0]))
    #         if self.option_type == "call":
    #             dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(self.payoff.x.reshape(1,-1)-dt_pde["K"]))
    #         else:
    #             dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(dt_pde["K"]-self.payoff.x.reshape(1,-1)))
            
    #         for _k in dt_pde.keys():
    #             if _k not in res.keys():
    #                 res[_k] = []
    #             res[_k].append(dt_pde[_k].clone())
    #         res["payoff"].append(dt_payoff)

    #         xs = self.pde.sde(torch.full_like(dt_pde["t"], self.T/self.num_ex), dt_pde["x"], dt_pde["r"], dt_pde["q"], dt_pde["sigma"])
    #         print(f"The device of dt_pde.t is {device}.")

    #         res["y"].append(
    #             torch.exp(- dt_pde["r"] * self.T/self.num_ex) * parallel_interpolation(xs, self.payoff.x, dt_payoff, interp_method=self.interp_method)
    #         )
    #     for _k in res.keys():
    #         res[_k] = torch.concat(res[_k], dim=0)
    #     return res


class Data_Saved(Dataset):
    def __init__(self, path, n_batches=None, iter_all=True):
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.data_dir = path
        self.file_list = sorted(os.listdir(self.data_dir))
        self._idx = 0
        if n_batches is None:
            self.n_batches = len(self.file_list)
        else:
            self.n_batches = n_batches
        self.iter_all = iter_all

    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[self._idx])
        data = torch.load(file_path, map_location=self.device)
        self._idx += 1
        self._idx = 0 if self._idx == len(self.file_list) else self._idx
        if not self.iter_all:
            self._idx = 0 if self._idx == self.n_batches else self._idx
        return data

class Bermudan(ABC):
    def __init__(self, pde, payoff, config):
        super().__init__()
        self.pde = pde
        self.payoff = payoff
        self.sensor = self.payoff.x
        self.T = config["T"]
        self.num_ex = config["num_ex"]
        self.option_type = config["option_type"]
        self.output_params = config["output_params"]

    def dataloader(self, batch_size, n_batches, frezed_params, interp_method, var_rescale=False, var_rescale_k=1, device="cpu"):
        return DataLoader(
            Data_Bermudan(self.pde, self.payoff, self.T, self.num_ex, self.option_type, batch_size, n_batches, frezed_params, interp_method, var_rescale, var_rescale_k, device), None
        )
    
    @abstractmethod
    def solution(self, batch):
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class Bermudan_1D(Bermudan):
    def __init__(self, pde, payoff, config):
        super().__init__(pde, payoff, config)

    def solution(self, batch):
        N = int(batch["x"].shape[0] / self.num_ex)
        batch_x = batch["x"].clone() # store the batch["x"]
        batch = {
            _param: batch[_param][:N] for _param in batch.keys()
        }
        
        grid, values = CN_bermudan_1D(cpflag=self.option_type, K=batch["K"].flatten(), T=self.T, num_ex=self.num_ex, 
                        vol=batch["sigma"].flatten(), r=batch["r"].flatten(), d=batch["q"].flatten())
        
        res = [parallel_interpolation(batch_x[N*i:N*(i+1),:], grid, values[i], interp_method="linear") for i in range(self.num_ex)]
        return torch.concat(res, dim=-1)

class Bermudan_basket(Bermudan):
    def __init__(self, pde, payoff, config):
        super().__init__(pde, payoff, config)

    def solution(self, batch):
        pass


BERMUDANS = {bermudan.__name__: bermudan for bermudan in Bermudan.get_subclasses()}

