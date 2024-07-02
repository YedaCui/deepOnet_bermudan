from deep_kolmogorov import trainer
import torch
from deep_kolmogorov import utils

config = {
        "seed": 0,
        "checkpoint": True,
        "pde": "BSr",
        "net": "DeepONet",
        "payoff": "GRF",
        "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
        "size_sensor": 50,
        "kernel": "RBF", 
        "length_scale":10,
        "var_scale":1,
        "bermudan": "Bermudan_1D",
        "T": 1,
        "num_ex": 2,
        "option_type": "put",
        "output_params": ["x", "r", "q", "sigma", "K"],
        "frezed_params": {"t":0},
        "interp_method": "linear",
        "opt": "adamw",
        "bs_train": 10000,
        "bs_test": 100,
        # "n_train_batches": 2000,
        # "n_test_batches": 1000,
        "n_train_batches": 3,
        "n_test_batches": 3,
        "accu_steps": 1,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_MP_num_sensor_50",
        "n_iterations": 30,
        "size_t_x_u": [0,1,4],
        "gpus":0,
        "num_width" : 35,
        "num_depth" : 5,
    }

mytrainer = trainer.Trainer(config)


for i in range(10):
    mytrainer._iteration += 1
    mytrainer.step()
