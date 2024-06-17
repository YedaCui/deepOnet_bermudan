import torch
import copy
from torch.utils.data import Dataset, DataLoader
import utils


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
        res = {"payoff": [],
               "y": []
               }
        for i in range(self.num_ex):
            dt_pde["t"].fill_(i*self.T/self.num_ex)
            dt_pde["x"] = self.pde.get_X(dt_pde)
            dt_pde["t"] += self.T/self.num_ex
            if i == self.num_ex-1:
                cont_value = torch.zeros(1,1, device=dt_pde["t"].device)
            else:
                cont_value = self.pde.option_price(self.T - dt_pde["t"], self.payoff.x.T, dt_pde["sigma"], dt_pde["r"], dt_pde["K"], option_type=self.option_type)
                cont_value += self.payoff.random(cont_value.shape[0])
            if self.option_type == "call":
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(self.payoff.x.T-dt_pde["K"]))
            else:
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(dt_pde["K"]-self.payoff.x.T))
            
            for _k in dt_pde.keys():
                if _k not in res.keys():
                    res[_k] = []
                res[_k].append(dt_pde[_k].clone())
            res["payoff"].append(dt_payoff)

            xs = self.pde.sde(torch.full_like(dt_pde["t"], self.T/self.num_ex), dt_pde["x"], dt_pde["r"], dt_pde["sigma"])

            res["y"].append(
                torch.exp(- dt_pde["r"] * self.T/self.num_ex) * utils.parallel_interpolation(xs, self.payoff.x, dt_payoff, interp_method=self.interp_method)
            )
        for _k in res.keys():
            res[_k] = torch.concat(res[_k], dim=0)
        return res


class Data_Saved(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
    
    def __getitem__(self, idx):
        return {
            _param: self.data[_param][idx] for _param in self.data.keys()
        }

class Bermudan():
    def __init__(self, pde, payoff, config):
        self.pde = pde
        self.payoff = payoff
        self.T = config["T"]
        self.num_ex = config["num_ex"]
        self.option_type = config["option_type"]
        self.output_params = config["output_params"]

    def dataloader(self, batch_size, n_batches, frezed_params, interp_method):
        return DataLoader(
            Data_Bermudan(self.pde, self.payoff, self.T, self.num_ex, self.option_type, batch_size, n_batches, frezed_params, interp_method), None
        )

    def solution(self, batch):
        return self.pde.option_price(self.T, batch["x"], batch["sigma"], batch["r"], batch["K"], option_type=self.option_type)
        

