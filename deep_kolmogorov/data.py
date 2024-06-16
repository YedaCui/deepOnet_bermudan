import torch
import copy
from torch.utils.data import Dataset, DataLoader


class Data_Bermudan(Dataset):
    def __init__(self, pde, payoff, config):
        self.pde = pde
        self.payoff = payoff
        self.batch_size = config["batch_size"]
        self.n_batches = config["n_batches"]
        self.frezed_params = config["frezed_params"]
        self.T = config["T"]
        self.num_ex = config["num_ex"]
        self.option_type = config["option_type"]

    
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
            res["y"].append(
                torch.exp(- dt_pde["r"] * self.T/self.num_ex) * self.pde.sde(torch.full_like(dt_pde["t"], self.T/self.num_ex), dt_pde["x"], dt_pde["r"], dt_pde["sigma"], dt_pde["K"], option_type=self.option_type)
            )
        for _k in res.keys():
            res[_k] = torch.concat(res[_k], dim=0)
        return res











