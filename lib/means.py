import torch
import gpytorch

class NegativeQuadraticMean(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size()):
        super().__init__()

        self.register_parameter(name="m0", parameter=torch.nn.Parameter(torch.zeros((*batch_shape,1))))

        self.register_parameter(name="xm", parameter=torch.nn.Parameter(torch.zeros((*batch_shape, input_size))))

        self.register_parameter(name="raw_scale", parameter=torch.nn.Parameter(torch.empty((*batch_shape, input_size))))
        self.register_constraint("raw_scale", gpytorch.constraints.Positive())
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(torch.ones((*batch_shape, input_size))))

    def forward(self, x):
        batched_xm = self.xm.view((*self.xm.shape[:-1], *(1,)*(len(x.shape)-1), self.xm.shape[-1]))
        batched_scale = self.scale.view((*self.scale.shape[:-1], *(1,)*(len(x.shape)-1), self.scale.shape[-1]))

        return self.m0 - 0.5*torch.sum((x-batched_xm)**2/batched_scale**2, -1)

    @property
    def scale(self):
        return self.raw_scale_constraint.transform(self.raw_scale)

    @scale.setter
    def scale(self, value):
        self._set_scale(value)
        
    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
