import torch
import botorch

class MAXV(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(self, model):
        super().__init__(model=model)
    
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        r"""
            X: A `b1 x ... bk x 1 x d`-dim batched tensor of `d`-dim design points.
        Returns:
            A `b1 x ... bk`-dim tensor
        """
        posterior = self._get_posterior(X=X.to(self.model.train_inputs[0])).mvn
        return (2*posterior.mean + posterior.variance + torch.log(torch.exp(posterior.variance) - 1))[...,0].to(X)
    
class MAXIQR(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(self, model, u):
        assert u > 0
        super().__init__(model=model)
        self.u = torch.tensor(u).to(model.train_inputs[0])
        self.X_pending = None
        
    def set_X_pending(self, X_pending = None) -> None:
        if X_pending is not None:
            self.X_pending = X_pending.detach().clone()
        else:
            self.X_pending = X_pending

    @botorch.utils.transforms.concatenate_pending_points
    def forward(self, X):
        r"""
            X: A `b1 x ... bk x q x d`-dim batched tensor of `d`-dim design points.
        Returns:
            A `b1 x ... bk`-dim tensor
        """
        X = X.to(self.model.train_inputs[0])
        if self.X_pending is None:
            posterior = self.model(X)
            ust = self.u * posterior.stddev
            acq = (posterior.mean + ust + torch.log(1-torch.exp(-2*ust)))[...,0]
#             print(1, acq)
        else:
            posterior = self.model(X)
            
            K = posterior.covariance_matrix[:,:1,1:]
            K_new = posterior.lazy_covariance_matrix[:,1:,1:].add_diag(self.model.likelihood.noise)
            
            t = K_new.inv_matmul(left_tensor=K, right_tensor=K.transpose(-1,-2))
            
            ust = self.u * torch.sqrt(
                torch.maximum(posterior.variance[...,0] - t[...,0,0], torch.tensor(1e-6))
            )
            
            acq = posterior.mean[...,0] + ust + torch.log(1-torch.exp(-2*ust))
#             print(2, acq)
        return acq