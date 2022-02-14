import botorch
import gpytorch
import torch

from . import means


class LogLikelihoodModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    def __init__(self, train_x, train_y, kernel_maker, batch_shape=torch.Size()):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)
        super().__init__(train_x, train_y, likelihood)
        self.mean = means.NegativeQuadraticMean(train_x.shape[-1], batch_shape=batch_shape)

        self.kernel_maker = kernel_maker
        self.kernel = kernel_maker(batch_shape=batch_shape)

        self.parameters_batch_shape = batch_shape
        self._num_outputs = 1

        self.parameter_distributions = {}

    def register_parameter_distribution(self, name, prior, setting_closure, getter_closure):
        self.parameter_distributions[name] = (prior, setting_closure, getter_closure)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))

    def train_log_likelihood(self):
        self.train()
        self.likelihood.train()

        x = self.train_inputs[0]
        y = self.train_targets

        f = self(x)
        return self.likelihood(f).log_prob(y)
