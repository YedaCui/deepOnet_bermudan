
from torch.utils.data import Dataset, DataLoader


class Data_Bermudan(Dataset):
    def __init__(self, pde, payoff, batch_size, n_batches, frezed_params : dict = {}):
        self.pde = pde
        self.payoff = payoff
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.frezed_params = frezed_params

    
    def __len__(self):
        return self.n_batches
    
    def __getitem__(self, idx):
        '''
        idx : useless arg in our case.
        '''

        dt_pde = next(iter(self.pde.dataloader(self.batch_size, 1, 'train')))
        
