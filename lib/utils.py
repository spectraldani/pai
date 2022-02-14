import copy
import sys
import warnings
from types import SimpleNamespace

import gpytorch
import numpy as np
import torch


def rgetattr(o, k_list):
    if len(k_list) == 0:
        return o
    else:
        return rgetattr(getattr(o, k_list[0]), k_list[1:])


def print_module(model, n_digits=10):
    print(f'{"parameter":35} {"device":7} {"dtype":7} {"shape":7}')
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            param_name = param_name.replace('raw_', '')
            param = rgetattr(model, param_name.split("."))
            print(
                f'{param_name:35} {param.device.type:7} {str(param.dtype)[6:]:7} '
                f'{str(tuple(param.shape)):7} {param.numpy().round(n_digits)}'
            )


def sample_all_priors(model):
    for distribution, setting_closure, _ in model.parameter_distributions.values():
        setting_closure(model, distribution.sample())


def sample_prior_get_best(gp, mll, at_least_num_samples):
    initial_state_dict = copy.deepcopy(gp.state_dict())

    best_state_dict = None
    best_llk = torch.as_tensor(-np.inf)
    results = [None] * at_least_num_samples

    i = 0
    while i < at_least_num_samples:
        try:
            gp.load_state_dict(initial_state_dict)
            sample_all_priors(gp)
            with warnings.catch_warnings(record=True) as ws:
                #                 llk = gp.train_log_likelihood()
                llk = mll(gp(gp.train_inputs[0]), gp.train_targets).item()
            if len(ws) == 0:
                results[i] = SimpleNamespace(llk=llk, state_dict=copy.deepcopy(gp.state_dict()))
                if results[i].llk > best_llk:
                    best_llk = results[i].llk
                    best_state_dict = results[i].state_dict
                i += 1
            else:
                print('Failed', i, file=sys.stderr)
        except gpytorch.utils.errors.NotPSDError as e:
            print('Exception', i, e, file=sys.stderr)
    gp.load_state_dict(best_state_dict)
    results.sort(key=lambda x: -x.llk)
    return results
