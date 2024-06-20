from .pdes import HYPERCUBES, PDES
from .modeling import Metrics, KolmogorovNet, NETS, NORMLAYERS
from .bermudan import BERMUDANS
from .payoff import PAYOFFS
import torch
import numpy as np
import os

def sampling(config):
    path = config["path"]
    pde_kwargs = (
        {"hypercubes": HYPERCUBES[config["hypercubes"]]}
        if "hypercubes" in config
        else {}
    )
    pde = PDES[config["pde"]](**pde_kwargs)
    payoff_kwargs = {
        _arg: config[_arg] for _arg in ["sensor", "kernel", "length_scale"] if _arg in config.keys()
        }
    payoff = PAYOFFS[config["payoff"]](**payoff_kwargs)
    bermudan = BERMUDANS[config["bermudan"]](pde, payoff, config)

    train_loader = bermudan.dataloader(config["bs_train"], config["n_train_batches"], config["frezed_params"], config["interp_method"])
    test_loader = bermudan.dataloader(config["bs_val"], config["n_test_batches"], config["frezed_params"], config["interp_method"])
    val_loader = bermudan.dataloader(config["bs_test"], config["n_test_batches"], config["frezed_params"], config["interp_method"])
    
    # if not os.path.exists(os.path.join(path,"config.json")):
    #     os.makedirs(os.path.join(path,"config.json"))
    # with open(os.path.join(path,"config.json"), "w") as f:
    #     json.dump(config, f)

    def save_data(path, dt_type, dt_loader):
        path = os.path.join(path,dt_type)
        if not os.path.exists(path):
            os.makedirs(path)
        _idx = 0
        for batch in dt_loader:
            if dt_type in ["val", "test"]:
                batch["solution"] = bermudan.solution(batch)
            batch = {
                _param: torch.from_numpy(batch[_param]) if isinstance(batch[_param], np.ndarray) else batch[_param] 
                for _param in batch.keys()
            }
            
            torch.save(batch,os.path.join(path,f"train_{_idx}.pt"))
            _idx += 1

    save_data(path, "train", train_loader)
    save_data(path, "val", val_loader)
    save_data(path, "test", test_loader)
    

