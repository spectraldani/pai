import copy
import math
import warnings
from types import SimpleNamespace

import botorch
import gpytorch
import torch

import lib.distributions
import lib.gps
import lib.priors
from lib.utils import sample_prior_get_best


def kernel_maker(ard_num_dims):
    def f(batch_shape=None):
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    return f


def train_gp(gp, maxiter=1000, **kwargs):
    gp.cpu()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.train()
    result = botorch.optim.fit.fit_gpytorch_scipy(
        mll,
        options=dict(maxiter=maxiter, **kwargs)
    )[1]['OptimizeResult']
    llk = mll(gp(gp.train_inputs[0]), gp.train_targets).item()
    return [SimpleNamespace(ok=result.success, llk=llk, state_dict=copy.deepcopy(gp.state_dict()),
                            optimization_result=result)]


def multi_start_train(gp, num_starts, num_samples, maxiter=1000, try_initial=False, initial_maxiter=None, **kwargs):
    previous_state_dict = copy.deepcopy(gp.state_dict())
    results: SimpleNamespace = [None] * num_starts

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.train()
    for i in range(num_starts):
        result_samples = sample_prior_get_best(gp, mll, num_samples)
        initial_state_dict = copy.deepcopy(gp.state_dict())
        with warnings.catch_warnings(record=True) as ws:
            _, optimization_result = botorch.optim.fit.fit_gpytorch_scipy(
                mll, options=dict(maxiter=maxiter, **kwargs)
            )
            optimization_result = optimization_result['OptimizeResult']
            try:
                llk = mll(gp(gp.train_inputs[0]), gp.train_targets).item()
            except gpytorch.utils.errors.NotPSDError:
                llk = -math.inf
        results[i] = SimpleNamespace(
            llk=llk,
            ok=optimization_result.success and len(ws) == 0 and llk > -math.inf,
            optimization_result=optimization_result,
            warnings=ws,
            state_dict=copy.deepcopy(gp.state_dict()),
            initial_state_dict=initial_state_dict,
        )
    if try_initial:
        gp.load_state_dict(copy.deepcopy(previous_state_dict))

        if initial_maxiter is None:
            initial_maxiter = maxiter
        with warnings.catch_warnings(record=True) as ws:
            _, optimization_result = botorch.optim.fit.fit_gpytorch_scipy(
                mll, options=dict(maxiter=initial_maxiter, **kwargs)
            )
            optimization_result = optimization_result['OptimizeResult']
            try:
                llk = mll(gp(gp.train_inputs[0]), gp.train_targets).item()
            except gpytorch.utils.errors.NotPSDError:
                llk = -math.inf
        results.append(SimpleNamespace(
            llk=llk,
            ok=optimization_result.success and len(ws) == 0 and llk > -math.inf,
            optimization_result=optimization_result,
            warnings=ws,
            state_dict=copy.deepcopy(gp.state_dict()),
            initial_state_dict=previous_state_dict,
        ))

    results.sort(key=lambda x: (not x.ok, -x.llk))
    gp.load_state_dict(results[0].state_dict)
    return results


def initialize_model(gp, bounds):
    x = gp.train_inputs[0]
    y = gp.train_targets

    x_std = x.std(dim=0)
    x_mean = x.mean(dim=0)
    x_mad = torch.median((x - x_mean).abs())

    y_max = torch.max(y)
    y_var = y.var()

    gp.kernel.base_kernel.lengthscale = x_mad
    gp.kernel.outputscale = y_var
    gp.likelihood.noise_covar.noise = 1e-3
    gp.mean.initialize(m0=y_max, xm=x[torch.argmax(y)], scale=x_std)

    gp.likelihood.noise_covar.raw_noise.requires_grad = False

    assign_parameter_dists(gp, bounds)


def assign_priors(gp, bounds):
    x = gp.train_inputs[0]
    y = gp.train_targets

    y_max = torch.max(y)
    y_min = torch.min(y)

    d = x.shape[1]
    L = bounds[1] - bounds[0]

    gp.kernel.base_kernel.register_prior(
        'lengthscale_prior',
        lib.priors.SoftplusLogNormal(torch.log(math.sqrt(d / 6) * L), torch.tensor(3 / 2 * math.log(10))),
        'raw_lengthscale',
    )

    gp.mean.register_prior(
        'm0_prior',
        gpytorch.priors.SmoothedBoxPrior(y_min, y_max, sigma=1),
        'm0'
    )

    gp.mean.register_prior(
        'scale_prior',
        lib.priors.SoftplusLogNormal(torch.log(math.sqrt(d / 6) * L), torch.tensor(3 / 2 * math.log(10))),
        'raw_scale'
    )

    gp.mean.register_prior(
        'xm_prior',
        gpytorch.priors.SmoothedBoxPrior(bounds[0], bounds[1], sigma=0.01),
        'xm'
    )


def assign_parameter_dists(gp: lib.gps.LogLikelihoodModel, bounds: torch.Tensor):
    x = gp.train_inputs[0]
    y = gp.train_targets

    y_max = torch.max(y)
    y_min = torch.min(y)

    d = x.shape[1]
    L = bounds[1] - bounds[0]

    # units: u, shape: (1,d)
    gp.register_parameter_distribution(
        'kernel.base_kernel.lengthscale',
        gpytorch.priors.NormalPrior(torch.log(math.sqrt(d / 6) * L), torch.tensor(1.)),
        lambda m, x: m.kernel.base_kernel._set_lengthscale(x.exp()),
        lambda m: m.kernel.base_kernel.lengthscale.log(),
    )

    # units: u**2, shape: ()
    gp.register_parameter_distribution(
        'kernel.outputscale',
        gpytorch.priors.NormalPrior(y.std().log().subtract(1), torch.tensor(1.)),
        lambda m, x: m.kernel._set_outputscale(x.exp()),
        lambda m: m.kernel.outputscale.log(),
    )

    # units: u, shape: (1)
    gp.register_parameter_distribution(
        'mean.m0',
        gpytorch.priors.SmoothedBoxPrior(y_min, y_max, sigma=0.01),
        lambda m, x: m.mean.initialize(m0=x),
        lambda m: m.mean.m0,
    )

    # units: u, shape: (d)
    gp.register_parameter_distribution(
        'mean.scale',
        gpytorch.priors.NormalPrior(torch.log(math.sqrt(d / 6) * L), torch.tensor(1.)),
        lambda m, x: m.mean._set_scale(x.exp()),
        lambda m: m.mean.scale.log(),
    )

    # units: u, shape: (d)
    gp.register_parameter_distribution(
        'mean.xm',
        gpytorch.priors.SmoothedBoxPrior(bounds[0], bounds[1], sigma=0.01),
        lambda m, x: m.mean.initialize(xm=x),
        lambda m: m.mean.xm,
    )
