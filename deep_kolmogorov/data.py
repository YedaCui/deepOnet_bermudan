import torch
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

    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        '''
        idx : useless arg in our case.
        '''
        
        dt_pde = next(iter(self.pde.dataloader(self.batch_size, 1, 'train', frezed_params=self.frezed_params))) # generate a batch pdes data
        for i in range(self.num_ex):
            dt_pde["t"].fill_(i*self.T/self.num_ex)
            dt_pde["x"] = self.pde.get_X(dt_pde)
            


        
        
        



