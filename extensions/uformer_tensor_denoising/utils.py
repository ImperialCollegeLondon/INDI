from abc import abstractmethod
from functools import partial, update_wrapper
from importlib import import_module

import hydra
import torch
from hydra.utils import get_class
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer


def instantiate(config, *args, is_func: bool = False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert "_target_" in config, "Config should have '_target_' for class instantiation."
    target = config["_target_"]

    if "_convert_" in config:
        config["_convert_"] = config["_convert_"].lower()

    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit(".", 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != "_target_"})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, _recursive_=False, **kwargs)


class BaseEnsemble:
    def __init__(self, base_model_config, models_n, isgan=False):
        assert models_n > 0, "Can't train an ensemble with 0 or fewer models"

        self.models_n = models_n
        self.isgan = isgan

        self.models = nn.ModuleList(modules=[instantiate(base_model_config) for _ in range(models_n)])
        self.current_model_idx = 0

        self.using_cuda = False
        self.cuda_device = None
        self.to_what = None
        self.training = True

    def parameters(self, recurse: bool = True):
        return self.models.parameters(recurse=recurse)

    def ensemble_parameters(self, recurse: bool = True):
        parameters = {}

        for i in range(self.models_n):
            pars = self.models[i].parameters(recurse=recurse)
            if isinstance(pars, dict):
                pars = {k: list(v) for k, v in pars.items()}
            else:
                pars = list(pars)
            parameters[f"models.{i}"] = pars

        return parameters

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.models.named_parameters()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.models.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.models.load_state_dict(state_dict, strict=strict)

    def cuda(self, device=None):
        self.using_cuda = True
        self.cuda_device = device
        self.models[self.current_model_idx].cuda(self.cuda_device)

        return self

    def to(self, *args, **kwargs):
        self.to_what = (args, kwargs)
        self.models[self.current_model_idx].to(*self.to_what[0], **self.to_what[1])

        return self

    def train(self, mode: bool = True):
        self.training = mode
        self.models[self.current_model_idx].train(mode)

        return self

    def eval(self):
        self.training = False
        self.models[self.current_model_idx].eval()

        return self

    def start_next_model(self):
        self.current_model_idx += 1

        if self.using_cuda:
            self.models[self.current_model_idx - 1].cpu()
            self.models[self.current_model_idx].cuda(self.cuda_device)

        if self.to_what is not None:
            self.models[self.current_model_idx].to(*self.to_what[0], **self.to_what[1])

        if self.training:
            self.models[self.current_model_idx].train()
        else:
            self.models[self.current_model_idx].eval()

    def __getattr__(self, item):
        return getattr(self.models[self.current_model_idx], item)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class EnsembleOptimiserWrapper:
    def __init__(self, params, optimiser, zero=False, optimiser_8bit=False):
        super().__init__()
        self.optimisers = []
        self.current_optimiser_idx = 0

        self.zero = zero
        self.optimiser_8bit = optimiser_8bit

        if zero:
            optimiser["params"] = None
            optimiser = dict(optimiser)
            optimiser.pop("_convert_")
            optimiser.pop("params")

            if optimiser_8bit:
                optimiser_class = get_class(optimiser["_target_"].replace("torch", "bitsandbytes"))
            else:
                optimiser_class = get_class(optimiser["_target_"])

            optimiser.pop("_target_")
            optimiser_kwargs = {k: v for k, v in optimiser.items()}

            for k, parameters in params.items():
                self.optimisers.append(
                    ZeroRedundancyOptimizer(parameters, optimizer_class=optimiser_class, **optimiser_kwargs)
                )
        else:
            if optimiser_8bit:
                optimiser["_target_"] = optimiser["_target_"].replace("torch", "bitsandbytes")

            for k, parameters in params.items():
                self.optimisers.append(instantiate(optimiser, params=parameters))

    def start_next_model(self):
        self.current_optimiser_idx += 1

    def state_dict(self, **kwargs):
        states = {}
        for i in range(len(self.optimisers)):
            if self.zero:
                self.optimisers[i].consolidate_state_dict()
            states[i] = self.optimisers[i].state_dict(**kwargs)
        return states

    def load_state_dict(self, state_dict, **kwargs):
        for k, sd in state_dict.items():
            self.optimisers[k].load_state_dict(sd, **kwargs)

    def __getattr__(self, item):
        return getattr(self.optimisers[self.current_optimiser_idx], item)


class EnsembleSchedulerWrapper:
    def __init__(self, scheduler, optimizer):
        super().__init__()
        self.schedulers = []
        self.current_scheduler_idx = 0

        for i, optimizer in enumerate(optimizer.optimisers):
            self.schedulers.append(instantiate(scheduler, optimizer=optimizer))

    def start_next_model(self):
        self.current_scheduler_idx += 1

    def state_dict(self, **kwargs):
        return {i: self.schedulers[i].state_dict(**kwargs) for i in range(len(self.schedulers))}

    def load_state_dict(self, state_dict, **kwargs):
        for k, sd in state_dict.items():
            self.schedulers[k].load_state_dict(sd, **kwargs)

    def __getattr__(self, item):
        return getattr(self.schedulers[self.current_scheduler_idx], item)


class IndependentEnsemble(BaseEnsemble):
    def get_model(self, index):
        if self.isgan:
            return self.models[index].generator
        else:
            return self.models[index]

    def __call__(self, x, *args, **kwargs):
        if self.training:
            model_output = self.get_model(self.current_model_idx)(x, *args, **kwargs)

            return model_output
        else:
            original_current_model_device = next(self.get_model(self.current_model_idx).parameters()).device

            # only move to cpu if there are multiple models
            if self.current_model_idx > 0:
                self.get_model(self.current_model_idx).cpu()

            model_outputs = []
            for i in range(self.current_model_idx + 1):
                original_device = next(self.get_model(i).parameters()).device
                self.get_model(i).to(original_current_model_device)
                model_outputs.append(self.get_model(i)(x, *args, **kwargs))
                self.get_model(i).to(original_device)

            self.get_model(self.current_model_idx).to(original_current_model_device)

            var, mean = torch.var_mean(torch.stack(model_outputs), dim=0)
            return mean
