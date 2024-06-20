import math
import time
import torch
from torch import nn
from typing import List, Tuple

EPSILON = 1e-08
NORMLAYERS = {
    "layernorm": torch.nn.LayerNorm,
    "batchnorm": nn.BatchNorm1d,
    "none": nn.Identity,
}


class BaseNet(torch.nn.Module):
    """
    Base class for different networks.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params_groups = [{"params": self.parameters()}]
        self.active_groups = []

    def unfreeze_only_active(self):
        for group in self.params_groups:
            for param in group["params"]:
                if group in self.active_groups:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def update_active_groups(self, iteration):
        idx = iteration // self.config["unfreeze_patience"]
        if idx < len(self.params_groups):
            if self.config["unfreeze"] == "single":
                self.active_groups = [self.params_groups[idx]]
            elif self.config["unfreeze"] == "sequential":
                self.active_groups = self.params_groups[: idx + 1]
            else:
                self.active_groups = self.params_groups
        else:
            self.active_groups = self.params_groups

    def decay_lr(self, iteration):
        if not (iteration + 1) % self.config["lr_decay_patience"]:
            for params_group in self.active_groups:
                if params_group["lr"] > self.config["min_lr"]:
                    params_group["lr"] *= self.config["lr_decay"]

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass


class DenseNet(nn.Module):
    """
    The feed forward neural network
    """

    def __init__(self, num_layers: List[int]):
        super(DenseNet, self).__init__()
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(num_layers[i],
                eps=1e-6,
                momentum=0.99)
            for i in range(len(num_layers)-1)])
            
        self.dense_layers = nn.ModuleList([nn.Linear(num_layers[i-1], num_layers[i])
                             for i in range(1, len(num_layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        for i in range(len(self.dense_layers)):
            x = self.bn_layers[i](x)
            x = self.dense_layers[i](x)
            x = torch.relu(x)
        return x


class DeepONet(BaseNet):
    """
    The deepOnet, The arguments are hidden layers of brunch and trunk net
    brunch_layer: The list of hidden sizes of trunk nets;
    trunk_layer: The list of hidden sizes of trunk nets
    """

    def __init__(self, config):
        super().__init__(config)
        self.size_sensor = config["size_sensor"]
        self.size_t, self.size_x, self.size_u = self.config["size_t_x_u"]
        self.branch = DenseNet([self.size_u + self.size_sensor] + [self.config["num_width"]] * self.config["num_depth"])
        self.trunk = DenseNet([self.size_t + self.size_x] + [self.config["num_width"]] * self.config["num_depth"])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        The input of state can be either 3-dim or 4-dim but once fixed a problem the
        dimension of the input tensor is fixed.
        """
        sensor_tensor, time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_sensor], tensor[:, self.size_sensor:self.size_sensor+self.size_t], tensor[:, self.size_sensor+self.size_t:self.size_sensor+self.size_t+self.size_x], tensor[:, self.size_sensor+self.size_t+self.size_x:]
        br = self.branch(torch.cat([sensor_tensor,u_tensor], -1))
        tr = self.trunk(torch.cat([time_tensor, state_tensor], -1))
        value = torch.sum(br * tr, dim=-1, keepdim=True)
        return value



class DenseOperator(nn.Module):
    def __init__(self, num_outputs):
        super(DenseOperator, self).__init__()
        self.num_outputs = num_outputs
        self.w = None
    
    def forward(self, x):
        flat_dim = x.shape[-2] * x.shape[-1]
        x = x.view(x.shape[0], flat_dim)
        if self.w == None:
            self.w = nn.Linear(flat_dim, self.num_outputs, device=x.device)
        x = nn.functional.relu(self.w(x))
        return x


class KernelOperator(DenseOperator):
    def __init__(self, in_channels, out_channels, kernel_size, num_outputs):
        super(KernelOperator, self).__init__(num_outputs)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        return super(KernelOperator, self).forward(x)
    

class DeepKernelONet(DeepONet):
    def __init__(self, config):
        self.num_para = config["size_t_x_u"][-1] # number of all parameters
        self.in_channels = config["in_channels"] # number of time inhomogeneoust parameters
        self.num_timepoints = config["num_timepoints"] # number of time points of the TI parameters
        self.num_outputs = config["num_outputs"] # the output dimension of the embedding net
        self.total_u = self.num_para - self.in_channels + self.num_timepoints * self.in_channels # the total dims of parameters 

        config["size_t_x_u"] = [config["size_t_x_u"][0], config["size_t_x_u"][1], self.num_para - self.in_channels + self.num_outputs] # update the size_t_x_u
        super().__init__(config)
        self.kernel = KernelOperator(config["in_channels"], config["out_channels"], config["kernel_size"], config["num_outputs"])
        # bin = self.kernel(torch.randn(1,2,20, device=torch.device("cuda"))) # add it when loading checkpoint with the input shape same as the experiment

    def forward(self, tensor: Tuple[torch.Tensor]) -> torch.Tensor:
        time_tensor, state_tensor, u_tensor = tensor[:, 0:self.size_t], tensor[:, self.size_t:-self.total_u], tensor[:, -self.total_u:]
        # embedding
        u_const, u_ti = u_tensor[:, 0:self.num_para-self.in_channels], u_tensor[:, self.num_para-self.in_channels:].reshape(u_tensor.shape[0], self.in_channels, -1)
        u_ti_after_embedding = self.kernel(u_ti)

        inputs_for_deeponet = torch.concat([time_tensor, state_tensor, u_const, u_ti_after_embedding], dim=1)
        return super().forward(inputs_for_deeponet)


NETS = {net.__name__: net for net in BaseNet.get_subclasses()}


class KolmogorovNet(torch.nn.Module):
    """
    DL Kolmogorov model.
    """

    def __init__(self, net, bermudan, saved_data):
        super().__init__()
        self.net = net
        self.bermudan = bermudan
        self.saved_data = saved_data

    def forward(self, batch, train=True):
        if batch["x"].ndim == 3:
            batch = {
            _k: _v.squeeze(0) for _k, _v in batch.items()
            }
        with torch.no_grad():
            if train:
                y = batch["y"]
            else:
                if self.saved_data:
                    y = batch["solution"] # shape batch["x"].shape[0] / self.bermudan.num_ex
                else:
                    y = self.bermudan.solution(batch)
            tensor = torch.concat(
                [batch["payoff"],
                 self.bermudan.pde.normalize_and_flatten(batch, self.bermudan.output_params)], dim = 1
            )
        if train:
            y_pred = self.net.forward(tensor)
        else:
            y_pred = self.price_bermudan(batch)
        return {"bermudan": y, "net": y_pred}
    
    def price_bermudan(self, batch):
        sensor = self.bermudan.payoff.x.to(batch["x"].device)
        N = int(batch["x"].shape[0] / self.bermudan.num_ex)
        batch = {
            _param: batch[_param][:N] for _param in batch.keys()
        }
        for i in range(self.bermudan.num_ex, 0, -1):
            if i == self.bermudan.num_ex:
                cont_value = torch.zeros(1,1, device=batch["t"].device)
            else:
                size_sensor = sensor.shape[0]
                dt_payoff_sensor = dt_payoff.repeat_interleave(size_sensor, dim=0)
                batch_sensor = {_param:batch[_param].repeat_interleave(size_sensor, dim=0) for _param in self.bermudan.output_params}
                batch_sensor["x"] = sensor.repeat(batch["x"].shape[0],1)
                with torch.no_grad():
                    tensor = torch.concat(
                        [dt_payoff_sensor,
                        self.bermudan.pde.normalize_and_flatten(batch_sensor, self.bermudan.output_params)], dim = 1
                    )
                    cont_value = self.net.forward(tensor).reshape(batch["x"].shape[0],size_sensor)
            if self.bermudan.option_type == "call":
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(sensor.reshape(1,-1) - batch["K"]))
            else:
                print("device of  cont_value :")
                print(cont_value.device)
                print("device of  K :")
                print(batch["K"].device)
                print("device of  payoff :")
                print(sensor.device)
                dt_payoff = torch.maximum(cont_value, torch.nn.ReLU()(batch["K"]-sensor.reshape(1,-1)))
        
        with torch.no_grad():
            tensor = torch.concat(
                [dt_payoff,
                self.bermudan.pde.normalize_and_flatten(batch, self.bermudan.output_params)], dim = 1
            )
            res = self.net.forward(tensor)
        return res


        

    def test_greeks(self, batch, greeks=["delta"], method="autodiff", d=0.001):
        device = next(self.net.parameters()).device  # 获取模型所在的设备
        batch = {k: v.to(device) for k, v in batch.items()}  # 将 batch 中的所有数据移动到相同设备
        y = self.pde.get_greeks(batch, greeks)
        y_pred = self.net_greeks(batch, greeks, method, d=d)
        return {"pde": y, "net": y_pred}
    
    def net_greeks(self, batch, greeks, method, d=0.001):
        dict_greek_param = {"delta": "x", "vega": "sigma", "c-delta": "rho"}
        # if method == "autodiff":
        #     dict_param_len = {_v: batch[_v].shape[-1] for _v in self.pde.params}

        #     tensor = self.pde.normalize_and_flatten(batch)
        #     tensor.requires_grad = True
        #     stds = self.pde.normalize_and_flatten(batch,report_std=True)
        #     y = self.net(tensor)
        #     y.backward(torch.ones_like(y))
        #     tensor_grad = tensor.grad
        #     dict_param_ind = {
        #         _g:  sum([dict_param_len[_v] for _v in self.pde.params if _v != dict_greek_param[_g]]) for _g in greeks
        #     }
        #     return {
        #         _g: tensor_grad[:,[dict_param_ind[_g]]] / stds[dict_greek_param[_g]] for _g in greeks
        #     }
        if method == "autodiff":
            res = {}
            for _g in greeks:
                _param = dict_greek_param[_g]
                original_param = batch[_param].clone()
                # batch[_param] = batch[_param].clone().detach().requires_grad_(True)
                # batch_flat = self.pde.normalize_and_flatten(batch)
                # self.net.eval()
                # pred = self.net(batch_flat)
                # pred.backward(torch.ones_like(batch[_param]))
                # res[_g] = batch[_param].grad
                # batch[_param] = original_param

                res[_g] = original_param.clone()
                for _i in range(original_param.shape[-1]):
                    batch[_param] = original_param.clone()
                    param_needs_g = original_param.clone().detach()[:,[_i]].requires_grad_(True)
                    batch[_param] = torch.concat([original_param.clone().detach()[:,:_i], 
                                                        param_needs_g,
                                                        original_param.clone().detach()[:,(_i+1):]], dim=1)
                    batch_flat = self.pde.normalize_and_flatten(batch)
                    self.net.eval()
                    pred = self.net(batch_flat)
                    pred.backward(torch.ones_like(param_needs_g))
                    res[_g][:,[_i]] = param_needs_g.grad
            return res
        if method == "finidiff":
            res = {}
            for _g in greeks:
                _param = dict_greek_param[_g]
                original_param = batch[_param].clone()
                res[_g] = original_param.clone()
                for _i in range(original_param.shape[-1]):
                    batch[_param] = original_param.clone()
                    # _d = original_param.clone()[:,_i]*d # relative step size
                    _d = d # constant step size
                    with torch.no_grad():
                        if _param == "rho":
                            batch[_param][:,_i] = torch.min(original_param.clone()[:,_i] + _d, 0.8 * torch.ones(batch[_param][:,_i].shape, device=batch[_param].device)) # add upper bound for the \rho
                        else:
                            batch[_param][:,_i] = original_param.clone()[:,_i] + _d
                        xp = batch[_param][:,_i].clone()
                        tensor = self.pde.normalize_and_flatten(batch)
                        y_p = self.net.forward(tensor)

                        if _param == "rho":
                            batch[_param][:,_i] = torch.max(original_param.clone()[:,_i] - _d, -0.1 * torch.ones(batch[_param][:,_i].shape, device=batch[_param].device)) # add lower bound for the \rho
                        else:
                            batch[_param][:,_i] = original_param.clone()[:,_i] - _d
                        xm = batch[_param][:,_i].clone()
                        tensor = self.pde.normalize_and_flatten(batch)
                        y_m = self.net.forward(tensor)
                    res[_g][:,_i] = (y_p - y_m).flatten()/(xp-xm)
                batch[_param] = original_param
            return res
            



class Metrics:
    """
    Returns the metrics for our trainer.
    """

    names = ["mse", "L2^2", "mae", "L1", "L1_std"]

    def __init__(self):
        self.best = {name: 1.0e10 for name in self.names}
        self.last_improve = {name: 0 for name in self.names}
        self.t = 0.0
        self.steps = 0.0
        self._running = {metric: 0 for metric in self.names}
        self._count = 0
        self._current_t = time.time()

    def store(self, output, return_loss=None):
        abs_error = (output["bermudan"] - output["net"]).abs()
        magnitude = output["bermudan"].abs() + 1
        rel_error = abs_error / magnitude
        # rel_error = (abs_error / magnitude).mean(dim=1, keepdim=True) # component wisely relative error
        # rel_error = abs_error.sum(dim=1, keepdim=True) / ( output["pde"].abs().sum(dim=1, keepdim=True) + 1 ) # L1 relative error
        # rel_error = torch.sqrt((abs_error**2).sum(dim=1, keepdim=True)) / ( torch.sqrt((output["pde"]**2).sum(dim=1, keepdim=True)) + 1 ) # L2 relative error
        loss = {
            "mse": (abs_error ** 2).mean(),
            "L2^2": (rel_error ** 2).mean(),
            "mae": abs_error.mean(),
            "L1": rel_error.mean(),
            "L1_std": rel_error.std(),
        }
        for name in self.names:
            self._running[name] += loss[name].item()
        self._count += 1
        if return_loss:
            return loss[return_loss]

    def zero(self):
        self._running = {metric: 0 for metric in self._running}
        self._count = 0
        self._current_t = time.time()

    def finalize(self):
        current_t = time.time() - self._current_t
        self.t += current_t
        self.steps += self._count
        current = {
            "time": current_t,
            "steps": self._count,
            "overall time": self.t,
            "overall steps": self.steps,
        }
        current.update(
            {name: metr / self._count for name, metr in self._running.items()}
        )
        for name in self.names:
            if current[name] < self.best[name]:
                self.best[name] = current[name]
                self.last_improve[name] = 0
            else:
                self.last_improve[name] += 1
        current["L2"] = math.sqrt(current["L2^2"])
        return {
            "current": current,
            "best": self.best,
            "last improve": self.last_improve,
        }
