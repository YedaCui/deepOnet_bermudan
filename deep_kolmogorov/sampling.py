from .pdes import HYPERCUBES, PDES
from .modeling import Metrics, KolmogorovNet, NETS, NORMLAYERS
from .bermudan import BERMUDANS
from .payoff import PAYOFFS
import torch
import numpy as np
import os, json

def sampling(config, device="cuda", data_types=["train", "val", "test"]):
    path = config["data_path"]
    if not os.path.exists(path):
        os.makedirs(path)
    pde_kwargs = (
        {"hypercubes": HYPERCUBES[config["hypercubes"]]}
        if "hypercubes" in config
        else {}
    )
    pde = PDES[config["pde"]](**pde_kwargs)
    payoff_kwargs = {
        _arg: config[_arg] for _arg in ["sensor", "kernel", "length_scale", "var_scale"] if _arg in config.keys()
        }
    print(config["payoff"])
    payoff = PAYOFFS[config["payoff"]](**payoff_kwargs)
    bermudan = BERMUDANS[config["bermudan"]](pde, payoff, config)

    dt_loaders = {
        _data_type: bermudan.dataloader(config[f"bs_{_data_type}"], config[f"n_{_data_type}_batches"], config["frezed_params"], config["interp_method"],config["var_rescale"], config["var_rescale_k"], device)
        for _data_type in data_types
    }

    with open(os.path.join(path,"config.json"), "w") as f:
        json.dump(
            {_k: _v.tolist() if isinstance(_v, torch.Tensor) else _v
                    for _k, _v in config.items()}
                    , f)

    def save_data(path, dt_type, dt_loader):
        path = os.path.join(path,dt_type)
        if not os.path.exists(path):
            os.makedirs(path)
        _idx = 0
        for batch in dt_loader:
            if dt_type in ["val", "test"]:
                # if torch.cuda.is_available():
                #     batch = {_k: _v.to(device) for _k, _v in batch.items()}
                batch["solution"] = bermudan.solution(batch)
            batch = {
                _param: torch.from_numpy(batch[_param]) if isinstance(batch[_param], np.ndarray) else batch[_param] 
                for _param in batch.keys()
            }
            
            torch.save(batch,os.path.join(path,f"{dt_type}_{_idx}.pt"))
            _idx += 1

    for _data_type in data_types:
        save_data(path, _data_type, dt_loaders[_data_type])


