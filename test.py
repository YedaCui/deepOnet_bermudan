from deep_kolmogorov import trainer
import torch

config = {
        "seed": tune.grid_search([0]),
        "checkpoint": True,
        "pde": "BSr",
        "net": "DeepONet",
        "payoff": "GRF",
        "sensor": torch.exp(torch.linspace(-4, 5, 100)),
        "kernel": "RBF", 
        "length_scale":10,
        "T": 1,
        "num_ex": 1,
        "option_type": "put",
        "output_params": ["x"],
        "frezed_params": {"t":0, "r": 0.025, "sigma":0.3, "kappa": 1},
        "interp_method": "linear",
        "opt": "adamw",
        "bs": 120000,
        "lr": 0.01,
        "min_lr": 1e-8,
        "lr_decay": 0.25,
        "lr_decay_patience": 2,
        "weight_decay": 0.01,
        "unfreeze": "all",
        "unfreeze_patience": 1,
        "n_iterations": 30,
        "n_train_batches": 2000,
        "n_test_batches": 1,
        "size_t_x_u": [0,1,0],
        "size_sensor": 100,
        "num_width" : tune.grid_search([35,55,75]),
        "num_depth" : tune.grid_search([5,7]),
    }

mytrainer = trainer.Trainer(config)