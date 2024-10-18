from deep_kolmogorov import sampling
from deep_kolmogorov import utils
import torch

config = {
        "pde": "BSbasketGmean",
        "payoff": "GRF",
        "sensor": torch.from_numpy(utils.qmc_grid(n=800,d=3,seed=0)).float(),
        "grids": torch.from_numpy(utils.MP_grid(n=20)).float(),
        "size_sensor": 800,
        "kernel": "RBF",
        "length_scale": 10,
        "var_scale": 1,
        "var_rescale": False,
        "var_rescale_k": 1,
        "bermudan": "Bermudan_basket",
        "T": 1,
        "num_ex": 2,
        "option_type": "put",
        "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
        "frezed_params": {"t":0},
        "interp_method": "linear",
        "bs_train": 10000,
        "bs_test": 5,
        "bs_val": 5,
        "n_train_batches": 200,
        "n_test_batches": 20000,
        "n_val_batches": 20000,
        "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/test"
    }
sampling.sampling(config=config, device="cuda:1", data_types=["train"])
