import numpy as np
import pystan


def norm_logpdf(x, loc=0, scale=1):
    n = x.shape[0]
    constants = np.log(2 * np.pi) + 2 * np.log(scale)

    z_score = (x - loc) / scale

    return -0.5 * (np.square(z_score) + constants)


class StanDistribution:
    _stan_model = None
    _model_code = None
    _target_variable = None

    @classmethod
    def _compile_model(self):
        assert self._model_code is not None
        self._stan_model = pystan.StanModel(model_code=self._model_code, model_name=self.__name__)

    def logpdf(self, x):
        raise NotImplementedError()

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, sample_size=1000, seed=None, chains=4, permuted=True, return_logpdf=True, **kwargs):
        """Returns array with shape (sample_size, d+1)"""
        assert self._target_variable is not None
        if self._stan_model is None:
            self._compile_model()
        sampled = self._stan_model.sampling(data=self._data(), iter=sample_size * 2 // chains, chains=chains, seed=seed,
                                            **kwargs)

        samples = sampled.extract(permuted=permuted, inc_warmup=False, pars=[self._target_variable])
        samples = samples[self._target_variable].reshape(sample_size, -1)
        if not return_logpdf:
            return samples
        log_posterior = self.logpdf(samples)
        return np.concatenate(
            (samples, log_posterior.reshape(sample_size, 1)),
            axis=-1
        )


class PolynomialPosterior(StanDistribution):
    _model_code = r"""
        functions {
            real ep_normal_lpdf(real y, real a, real b, real C){
                return C * normal_lpdf(y | a, b);
            }
        }
        data {
            int n;
            real y[n];

            int degree;
            real coeffs[2, degree+1];
            real std;
            
            real C;
        }
        parameters {
            vector[2] theta;
        }
        transformed parameters {
            vector[2] poly;
            poly[1] = 0; poly[2] = 0;
            for (i in 1:(degree+1)){
                poly[1] = poly[1]*theta[1] + coeffs[1, i];
                poly[2] = poly[2]*theta[2] + coeffs[2, i];
            }
        }
        model {
            theta[1] ~ ep_normal(0,0.25,C);
            theta[2] ~ ep_normal(0,0.25,C);
            for (i in 1:n){
                target += log_sum_exp(
                    log(0.5) + normal_lpdf(y[i] | poly[1], std),
                    log(0.5) + normal_lpdf(y[i] | poly[2], std)
                );
            }
        }
    """

    _target_variable = 'theta'

    def _data(self):
        return dict(
            n=self.n, y=self.observed_data.reshape(-1),
            degree=self.max_degree,
            coeffs=[
                np.pad(p.coef, (0, self.max_degree - len(p.coef) + 1), constant_values=0)[::-1]
                for p in self.polys
            ],
            std=self.std, C=self.parallel_factor
        )

    def __init__(self, observed_data, roots, std=1, parallel_factor=1):
        super().__init__()
        self.observed_data = observed_data
        self.std = std
        self.parallel_factor = parallel_factor

        self.polys = [
            np.polynomial.Polynomial.fromroots(r)
            for r in roots
        ]
        self.max_degree = max(p.degree() for p in self.polys)

        self.n = len(observed_data)
        assert observed_data.shape == (self.n, 1)
        assert 0 < parallel_factor <= 1
        assert std > 0

    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 2, f'x must be 2d: x.shape={x.shape}, d={x.shape[-1]}'

        # (m, 2)
        prior = (
                    # scipy.stats.uniform.logpdf(x, loc=-1, scale=2) # Prior
                    norm_logpdf(x, scale=0.25)
                ) * self.parallel_factor

        # (n,m)
        likelihood = np.logaddexp(
            np.log(0.5) + norm_logpdf(self.observed_data, loc=self.polys[0](x[:, 0]), scale=self.std),
            np.log(0.5) + norm_logpdf(self.observed_data, loc=self.polys[1](x[:, 1]), scale=self.std)
        )

        return likelihood.sum(0) + prior.sum(1)
