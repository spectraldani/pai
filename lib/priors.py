import torch
from gpytorch.priors import Prior, NormalPrior
from gpytorch.priors.utils import _bufferize_attributes
from torch.distributions import StudentT
from torch.nn import Module
from torch.nn.functional import softplus


class StudentTPrior(Prior, StudentT):
    def __init__(self, df, loc, scale, validate_args=False, transform=None):
        Module.__init__(self)
        StudentT.__init__(self, df=df, loc=loc, scale=scale, validate_args=validate_args)
        _bufferize_attributes(self, ('df', 'loc', 'scale'))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return StudentTPrior(self.df.expand(batch_shape), self.loc.expand(batch_shape), self.scale.expand(batch_shape))


class SoftplusLogNormal(NormalPrior):
    def __init__(self, loc, scale, lower_bound=0, validate_args=False):
        super().__init__(loc=loc, scale=scale, validate_args=validate_args, transform=None)
        self.lower_bound = torch.as_tensor(lower_bound).to(loc)

    def log_prob(self, r):
        sp_r = torch.maximum(softplus(r), torch.as_tensor(torch.finfo(r.dtype).eps, dtype=r.dtype))
        x = sp_r + self.lower_bound
        log_x = x.log()
        return super().log_prob(log_x) + r - log_x - sp_r

    def rsample(self, sample_shape=torch.Size([])):
        return super().rsample(sample_shape=sample_shape).exp().sub(self.lower_bound).expm1().log()

    def sample(self, sample_shape=torch.Size([])):
        return super().sample(sample_shape=sample_shape).exp().sub(self.lower_bound).expm1().log()
