from deep_kolmogorov import trainer
import torch
from deep_kolmogorov import utils

# config = {
#         "seed": 0,
#         "checkpoint": True,
#         "pde": "BSr",
#         "net": "DNNKernel",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF", 
#         "length_scale":10,
#         "var_scale":1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x", "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "opt": "adamw",
#         "bs_train": 10000,
#         "bs_test": 100,
#         # "n_train_batches": 2000,
#         # "n_test_batches": 1000,
#         "n_train_batches": 2000,
#         "n_test_batches": 10,
#         "accu_steps": 1,
#         "lr": 0.01,
#         "min_lr": 1e-8,
#         "lr_decay": 0.25,
#         "lr_decay_patience": 4,
#         "weight_decay": 0.01,
#         "unfreeze": "all",
#         "unfreeze_patience": 1,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2",
#         "n_iterations": 30,
#         "size_t_x_u": [0,1,4],
#         "num_width" : 35,
#         "num_depth" : 5,
#         "num_outputs": 20, # the output dimension of the embedding net
#         "out_channels": 5,
#         "kernel_size": 20,
#         "gpus":0,
#     }

# mytrainer = trainer.Trainer(config)


# for i in range(10):
#     mytrainer._iteration += 1
#     mytrainer.step()


# import torch
# from deep_kolmogorov import pdes, lsmc


# pde = pdes.BSbasketGmean()
# d = 5
# regmethod = lsmc.PolynomialReg(2,d)
# T = 1
# num_ex = 4
# num_sim = 1000000

# pricer = lsmc.LSMC(pde, regmethod, T, "call", num_ex, num_sim)

# batch= {# "s": torch.ones(1,d)*100,
#         "s": torch.tensor([[100,150,200,175, 125]]).float(),
#         # "sigma": torch.ones(1,d)*0.2,
#         "sigma": torch.tensor([[0.2,0.3,0.25,0.24,0.15]]).float(),
#         "t": torch.ones(1,1)*0.0,
#         "r": torch.ones(1,1)*0.01,
#         # "q": torch.ones(1,1)*0.1,
#         "q": torch.tensor([[0.03,0.02,0.05,0.0,0.04]]).float(),
#         "rho": torch.ones(1,1)*0.3,
#         }
# batch["K"] = torch.mean(batch["s"],dim=-1, keepdim=True)

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