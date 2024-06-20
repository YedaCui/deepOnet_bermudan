from deep_kolmogorov import sampling
import torch

config = {
        "pde": "BSr",
        "payoff": "GRF",
        "sensor": torch.exp(torch.linspace(-4, 5, 100)),
        "kernel": "RBF",
        "length_scale":10,
        "bermudan": "Bermudan_1D",
        "T": 1,
        "num_ex": 1,
        "option_type": "put",
        "output_params": ["x"],
        "frezed_params": {"t":0, "r": 0.025, "q":0.05, "sigma":0.3, "kappa": 1},
        "interp_method": "linear",
        "bs_train": 10000,
        "bs_test": 100,
        "n_train_batches": 1,
        "n_test_batches": 1000,
        "size_sensor": 100,
        "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/testing"
    }
sampling.sampling(config=config)