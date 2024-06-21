import torch
from torch.utils.data import Dataset, DataLoader
from .utils import *
import os
import ray


class Data_Bermudan(Dataset):
    def __init__(self, pde, payoff, T, num_ex, option_type, batch_size, n_batches, frezed_params, interp_method):
        self.pde = pde
        self.payoff = payoff
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.frezed_params = frezed_params
        self.T = T
        self.num_ex = num_ex
        self.option_type = option_type
        self.interp_method = interp_method

    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        '''
        idx : useless arg in our case.
        '''

        dt_pde = next(iter(self.pde.dataloader(self.batch_size, 1, 'train', frezed_params=self.frezed_params))) # generate a batch pdes data
        device = dt_pde["t"].device
        res = {"payoff": [],
               "y": []
               }
        for i in range(self.num_ex):
            dt_pde["t"].fill_(i*self.T/self.num_ex)
            dt_pde["x"] = self.pde.get_X(dt_pde)
            dt_pde["t"] += self.T/self.num_ex
            if i == self.num_ex-1:
                cont_value = torch.zeros(1,1, device=device)
            else:
                cont_value = self.pde.option_price(self.T - dt_pde["t"], self.payoff.x.reshape(1,-1), dt_pde["sigma"], dt_pde["r"], dt_pde["q"], dt_pde["K"], option_type=self.option_type)
                cont_value += torch.from_numpy(self.payoff.random(cont_value.shape[0]))
            if self.option_type == "call":
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(self.payoff.x.reshape(1,-1)-dt_pde["K"]))
            else:
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(dt_pde["K"]-self.payoff.x.reshape(1,-1)))
            
            for _k in dt_pde.keys():
                if _k not in res.keys():
                    res[_k] = []
                res[_k].append(dt_pde[_k].clone())
            res["payoff"].append(dt_payoff)

            xs = self.pde.sde(torch.full_like(dt_pde["t"], self.T/self.num_ex), dt_pde["x"], dt_pde["r"], dt_pde["q"], dt_pde["sigma"])
            print(f"The device of dt_pde.t is {device}.")

            res["y"].append(
                torch.exp(- dt_pde["r"] * self.T/self.num_ex) * parallel_interpolation(xs, self.payoff.x, dt_payoff, interp_method=self.interp_method)
            )
        for _k in res.keys():
            res[_k] = torch.concat(res[_k], dim=0)
        return res


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

class Bermudan():
    def __init__(self, pde, payoff, config):
        self.pde = pde
        self.payoff = payoff
        self.sensor = self.payoff.x
        self.T = config["T"]
        self.num_ex = config["num_ex"]
        self.option_type = config["option_type"]
        self.output_params = config["output_params"]

    def dataloader(self, batch_size, n_batches, frezed_params, interp_method):
        return DataLoader(
            Data_Bermudan(self.pde, self.payoff, self.T, self.num_ex, self.option_type, batch_size, n_batches, frezed_params, interp_method), None
        )

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
        batch = {
            _param: batch[_param][:N] for _param in batch.keys()
        }
        
        grid, values = CN_bermudan_1D(cpflag=self.option_type, K=batch["K"].flatten(), T=self.T, num_ex=self.num_ex, 
                        vol=batch["sigma"].flatten(), r=batch["r"].flatten(), d=batch["q"].flatten())

        return parallel_interpolation(batch["x"], grid, values, interp_method="linear")
        

        
        
BERMUDANS = {bermudan.__name__: bermudan for bermudan in Bermudan.get_subclasses()}

