from deep_kolmogorov import sampling
from deep_kolmogorov import utils
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
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 0.5,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_0.5"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 5,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_1_length_scale_5"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 50,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_1_length_scale_50"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "kernel": "RBF",
#         "length_scale": 100,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_max_4.4_var_scale_1_length_scale_100"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 200)),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 50)),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:0", data_types=["val", "test"])


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.linspace(0.01,80,100),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_100_uniform"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.linspace(0.01,80,200),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_200_uniform"
#     }
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.linspace(0.01,80,50),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_num_sensor_50_uniform"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": 10*torch.exp(torch.linspace(-2.1,2.1, 100)),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_expuni_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100)),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_MP_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": 10*torch.exp(torch.linspace(-2.1,2.1, 200)),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_expuni_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_MP_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": 10*torch.exp(torch.linspace(-2.1,2.1, 50)),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_expuni_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.05_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50,g1=50,g2=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100,g1=50,g2=50)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 52,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_52_qmax_0.05_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 100)),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 200)),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 3*2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.exp(torch.linspace(-4, 4.4, 50)),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:0")



# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50,g1=50,g2=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:0", data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100,g1=50,g2=50)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2", data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 2000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.05_sensor_type_MP_num_sensor_200_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["val", "test"])



# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.05_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["train"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.05_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2", data_types=["train"])


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50,g1=50,g2=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100,g1=50,g2=50)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         # "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/testing"
#     }
# sampling.sampling(config=config, device="cuda:2")


# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
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
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_2_qmax_0.1_sensor_type_MP_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_50"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_100"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_4_qmax_0.1_sensor_type_MP_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50,g1=50,g2=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=50,g1=50,g2=50)).float(),
#         "size_sensor": 50,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_50_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:2", data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100,g1=50,g2=50)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.1_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=100,g1=50,g2=50)).float(),
#         "size_sensor": 100,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_100_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["val", "test"])

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 2000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_12_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:0")

# config = {
#         "pde": "BSr",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.MP_grid(n=200,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": True,
#         "var_rescale_k": 2,
#         "bermudan": "Bermudan_1D",
#         "T": 1,
#         "num_ex": 250,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 1000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 20000,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data/free_test_num_ex_250_qmax_0.1_sensor_type_MP_num_sensor_200_var_rescale_2"
#     }
# sampling.sampling(config=config, device="cuda:0", data_types=["val", "test"])




#######################################################################################################################################################
####################################################### generate data for geometric basket case #######################################################
#######################################################################################################################################################

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=200,d=3,seed=0)).float(),
#         "grids": torch.from_numpy(utils.MP_grid(n=20)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 200,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_2_qmax_0.1_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")

config = {
        "pde": "BSbasketGmean",
        "payoff": "GRF",
        "sensor": torch.from_numpy(utils.qmc_grid(n=400,d=3,seed=0)).float(),
        "grids": torch.from_numpy(utils.MP_grid(n=20)).float(),
        "size_sensor": 400,
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
        "bs_test": 100,
        "bs_val": 100,
        "n_train_batches": 200,
        "n_test_batches": 1000,
        "n_val_batches": 1000,
        "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_2_qmax_0.1_num_sensor_400"
    }
sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=800,d=3,seed=0)).float(),
#         "grids": torch.from_numpy(utils.MP_grid(n=20)).float(),
#         "size_sensor": 800,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 2,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 100,
#         "bs_val": 100,
#         "n_train_batches": 200,
#         "n_test_batches": 1000,
#         "n_val_batches": 1000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_2_qmax_0.1_num_sensor_800"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["test"])

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=200,d=3)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_4_qmax_0.1_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=400,d=3)).float(),
#         "size_sensor": 400,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_4_qmax_0.1_num_sensor_400"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=800,d=3)).float(),
#         "size_sensor": 800,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 4,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_4_qmax_0.1_num_sensor_800"
#     }
# sampling.sampling(config=config, device="cuda:1")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=200,d=3,g1=50,g2=50)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_12_qmax_0.1_num_sensor_200"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=400,d=3,g1=50,g2=50)).float(),
#         "size_sensor": 400,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_12_qmax_0.1_num_sensor_400"
#     }
# sampling.sampling(config=config, device="cuda:2")

# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=800,d=3,g1=50,g2=50)).float(),
#         "size_sensor": 800,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 12,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 200,
#         "n_test_batches": 10000,
#         "n_val_batches": 10000,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/free_test_num_ex_12_qmax_0.1_num_sensor_800"
#     }
# sampling.sampling(config=config, device="cuda:2")



# config = {
#         "pde": "BSbasketGmean",
#         "payoff": "GRF",
#         "sensor": torch.from_numpy(utils.qmc_grid(n=200,d=3,seed=0)).float(),
#         #  "grids": torch.from_numpy(utils.MP_grid(n=30)).float(),
#         "size_sensor": 200,
#         "kernel": "RBF",
#         "length_scale": 10,
#         "var_scale": 1,
#         "var_rescale": False,
#         "var_rescale_k": 1,
#         "bermudan": "Bermudan_basket",
#         "T": 1,
#         "num_ex": 1,
#         "option_type": "put",
#         "output_params": ["x",  "r", "q", "sigma", "rho", "K"],
#         "frezed_params": {"t":0,
#                           "s":10,
#                           "r":0.025,
#                           "q":0,
#                           "sigma":0.3,
#                           "rho":0,
#                           "kappa":1},
#         "interp_method": "linear",
#         "bs_train": 10000,
#         "bs_test": 10,
#         "bs_val": 10,
#         "n_train_batches": 20,
#         "n_test_batches": 1,
#         "n_val_batches": 1,
#         "data_path": "/home/ycui/Documents/deepOnet_bermudan/data_GeometricBasket/test1"
#     }
# sampling.sampling(config=config, device="cuda:1", data_types=["train"])