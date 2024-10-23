from deep_kolmogorov import trainer
import torch
from deep_kolmogorov import utils

# config = {
#         "seed": 0,
#         "checkpoint": True,
#         "pde": "BSbasketGmean",
#         "net": "DNN",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=800, d=3)).float(),
#         "size_sensor": 800,
#         "kernel": "RBF", 
#         "length_scale":10,
#         "var_scale":1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x", "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "opt": "adamw",
#         "bs_train": 10000,
#         "bs_test": 5,
#         "n_train_batches": 2,
#         "n_test_batches": 1,
#         "accu_steps": 1,
#         "lr": 0.01,
#         "min_lr": 1e-8,
#         "lr_decay": 0.25,
#         "lr_decay_patience": 2,
#         "weight_decay": 0.01,
#         "unfreeze": "all",
#         "unfreeze_patience": 1,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_2_qmax_0.1_num_sensor_800",
#         "n_iterations": 30,
#         "size_t_x_u": [0,3,9],
#         "num_width" : 35,
#         "num_depth" : 5,
#         "gpus":0,
#     }

# mytrainer = trainer.Trainer(config)
# mytrainer.step()


# for i in range(10):
#     mytrainer._iteration += 1
#     mytrainer.step()


# import torch
# from deep_kolmogorov import pdes, lsmc

# d=2
# pde = pdes.BSbasketGmean(d=d)
# regmethod = lsmc.PolynomialReg(2,d)
# T = 1
# num_ex = 10
# num_sim = 500000 * 10

# pricer = lsmc.LSMC(pde, regmethod, T, "put", num_ex, num_sim)

# batch= {# "s": torch.ones(1,d)*100,
#         "x": torch.tensor([[90,110]]).float(),
#         # "sigma": torch.ones(1,d)*0.2,
#         "sigma": torch.tensor([[0.2,0.3]]).float(),
#         "t": torch.ones(1,1)*0.0,
#         "r": torch.ones(1,1)*0.04,
#         # "q": torch.ones(1,1)*0.1,
#         "q": torch.tensor([[0.00,0.00]]).float(),
#         "rho": torch.ones(1,1)*0.25,
#         }
# batch["K"] =  torch.mean(batch["x"],dim=-1, keepdim=True)
# print(batch["K"])

# print(pricer.pricing(batch))

# pde = pdes.BSbasketMax()
# d = 1
# regmethod = lsmc.PolynomialReg(3,d)
# T = 1
# num_ex = 2
# num_sim = 100000
# pde, regmethod, T, num_ex, num_sim
# pricer = lsmc.LSMC(pde, regmethod, T, "put", num_ex, num_sim)

# batch= {"s": torch.ones(1,d)*10,
#         "sigma": torch.ones(1,d)*0.3,
#         "t": torch.ones(1,1)*0.0,
#         "r": torch.ones(1,1)*0.025,
#         "q": torch.ones(1,1)*0.0,
#         "K": torch.ones(1,1)*10,
#         "rho": torch.zeros(1,1),
#         }

# print(pricer.pricing(batch))



import torch
from deep_kolmogorov import pdes
from deep_kolmogorov.bermudan import Bermudan_basket
from deep_kolmogorov.payoff import GRF

d=2
pde = pdes.BSbasketGmean(d=d)
payoff = GRF(sensor = torch.from_numpy(utils.qmc_grid(n=800, d=2)).float())
config = {
        "T" : 1,
        "num_ex" : 50,
        "option_type" : "call",
        "output_params" : ["x", "r", "q", "sigma", "rho", "K"]
}
bermudan = Bermudan_basket(pde, payoff, config)

batch= {# "s": torch.ones(1,d)*100,
        "x": torch.tensor([[100,95] for _ in range (config["num_ex"])]).float(),
        # "sigma": torch.ones(1,d)*0.2,
        "sigma": torch.tensor([[0.1,0.25] for _ in range (config["num_ex"])]).float(),
        "t": torch.ones(config["num_ex"],1)*0.0,
        "r": torch.ones(config["num_ex"],1)*0.05,
        # "q": torch.ones(1,1)*0.1,
        "q": torch.tensor([[0.04,0.04] for _ in range (config["num_ex"])]).float(),
        "rho": torch.ones(config["num_ex"],1)*0.5,
        "K": torch.ones(config["num_ex"],1)*100,
        }

print(bermudan.solution(batch))