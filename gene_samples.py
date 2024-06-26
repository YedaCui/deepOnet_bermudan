from deep_kolmogorov import sampling
import torch

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 1,
#         "option_type": "put",
#         "output_params": ["x"],
#         "frezed_params": {"t":0, "r": 0.025, "q":0.05, "sigma":0.3, "kappa": 1},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 30*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/frezed_test_with_num_ex_1"
#     }
# sampling.sampling(config=config, data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x"],
#         "frezed_params": {"t":0, "r": 0.025, "q":0.05, "sigma":0.3, "kappa": 1},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 30*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/frezed_test_with_num_ex_2"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 5,
#         "option_type": "put",
#         "output_params": ["x"],
#         "frezed_params": {"t":0, "r": 0.025, "q":0.05, "sigma":0.3, "kappa": 1},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 30*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/frezed_test_with_num_ex_5"
#     }
# sampling.sampling(config=config, device="cuda:2", data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 10,
#         "option_type": "put",
#         "output_params": ["x"],
#         "frezed_params": {"t":0, "r": 0.025, "q":0.05, "sigma":0.3, "kappa": 1},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/frezed_test_with_num_ex_10"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 30*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2"
#     }
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 5,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 5*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_5"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 5, 100)),
#         "kernel": "RBF",
#         "length_scale":10,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 10,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 5*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_10"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 3*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_2"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 3,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 3*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_3"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 5,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 3*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "size_sensor": 100,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_5"
#     }
# sampling.sampling(config=config, device="cuda:1")

config = {
        "pde": "BSr",
        "payoff": "GRF",
        "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
        "kernel": "RBF",
        "length_scale": 10,
        "var_scale": 0.5,
        "bermudan": "Bermudan_1D",
        "T": 1,
        "num_ex": 2,
        "option_type": "put",
        "output_params": ["x",  "r", "q", "sigma", "K"],
        "frezed_params": {"t":0},
        "interp_method": "linear",
        "bs_train": 10000,
        "bs_test": 100,
        "bs_val": 100,
        "n_train_batches": 3*2000,
        "n_test_batches": 1000,
        "n_val_batches": 1000,
        "size_sensor": 100,
        "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_0.5"
    }
sampling.sampling(config=config, device="cuda:2")