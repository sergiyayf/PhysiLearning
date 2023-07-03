import pymc as pm
import arviz as az
import numpy as np
from physilearning.tools.odemodel import ODEModel
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import pandas as pd


class ODEBayesianFitter:
    """A class for fitting an ODE model to data using Bayesian inference.

    :param ode: The ODE model to fit.
    :param data: The data to fit the model to.

    :ivar model: The PyMC4 model.
    :ivar data: The data to fit the model to.
    :ivar ode: The ODE model to fit.
    :ivar trace: The trace of the MCMC sampling.

    :method set_priors: Define the priors for the model parameters.
    :method set_likelihood: Define the likelihood for the model.
    :method fit: Fit the model to the data.
    :method plot_trace: Plot the trace of the MCMC sampling.

    """

    def __init__(self, ode: ODEModel = ODEModel(), data: pd.DataFrame = None):

        self.model = pm.Model()
        self.data = data
        self.ode = ode
        self.trace = None

    def set_priors(self, prior_dist: str = "normal") -> dict:
        """Define the priors for the model parameters.

        :param prior_dist: The distribution to use for the priors.
        :return: (dict) The priors for the model parameters.
        """
        priors_dict = self.ode.params
        out = {}
        if prior_dist == "normal":
            with pm.Model() as self.model:
                for key, value in priors_dict.items():
                    out[key] = pm.TruncatedNormal(key, mu=self.ode.params[key], sigma=self.ode.params[key]*0.05,
                                                  lower=0, initval=self.ode.params[key])

                out['sigma'] = pm.HalfNormal("sigma", 10)

        elif prior_dist == "uniform":
            with pm.Model() as self.model:
                for key, value in priors_dict.items():
                    out[key] = pm.Uniform(key, lower=0, upper=3, initval=self.ode.params[key])

                out['sigma'] = pm.HalfNormal("sigma", 10)

        else:
            raise NotImplementedError("Only normal priors are currently supported.")

        return out

    @staticmethod
    @as_op(itypes=[pt.dvector, pt.dvector, pt.ivector, pt.dvector], otypes=[pt.dmatrix])
    def pytensor_matrix_solve(y0: pt.dvector, times: pt.dvector,
                              treatment_schedule: pt.ivector,
                              theta: pt.dvector) -> pt.dmatrix:
        """Define the ODE with pytensor inputs and outputs.

        :param y0: The initial conditions for the ODE.
        :param times: The times to solve the ODE at.
        :param treatment_schedule: The treatment schedule for the ODE.
        :param theta: The parameters for the ODE.

        :return: The solution to the ODE.

        """
        return ODEModel(y0=y0, theta=theta, time=times, dt=1,
                        treatment_schedule=treatment_schedule).simulate()

    def likelihood(self, priors, pytensor_op):
        """Define the likelihood function for the model.

        Parameters
        ----------
        priors : dict
            The priors for the model parameters.

        Returns
        -------
        pymc Normal distribution
            The likelihood function for the model.
        """

        with self.model:
            # define the likelihood
            sigma = priors["sigma"]

            theta = pm.math.stack([priors[key] for key, value in priors.items() if key != "sigma"])

            ode_solution = pytensor_op(theta)

            likelihood = pm.Normal("likelihood", mu=ode_solution, sigma=sigma, observed=self.data[["y", "x"]].values)

        return likelihood

    def sample(self, pytensor_op: callable, sampler="DEMetropolis", chains=8, draws=3000, cores=1):
        """
        Fit the Lotka-Volterra model to the data using a Bayesian approach.

        Returns
        -------
        trace : pymc.backends.base.MultiTrace
            The trace of the MCMC sampling.
        """


        priors = self.set_priors(prior_dist='normal')
        self.likelihood(priors=priors, pytensor_op= pytensor_op)
        vars_list = list(self.model.values_to_rvs.keys())[:-1]
        print(vars_list)
        sampler = sampler
        chains = chains
        draws = draws
        with self.model:
            trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], tune=draws, draws=draws, chains=chains, cores=cores)
        trace = trace_DEM
        self.trace = trace
        return trace

    def plot_inference_trace(self, ax=None, treatment_schedule=None, num_samples=25,
                             title="DEM inference", consts=None, params=None, **kwargs):
        """
        Plot the trace of the MCMC sampling.

        Parameters
        ----------
        trace : pymc.MCMC.MCMC
            The MCMC object containing the results of the Bayesian fitting.
        varnames : list, optional
            The variables to plot.
        figsize : tuple, optional
            The size of the figure.
        lines : dict, optional
            The lines to plot on the trace.
        combined : bool, optional
            Whether to plot the traces of all the variables on the same axis.
        """
        cols = self.ode.params.keys()
        trace_df = az.extract(self.trace, num_samples=num_samples).to_dataframe()
        #self.time = np.arange(1900, 1921, 0.01)
        for row_idx in range(num_samples):
            self.ode = ODEModel(theta=trace_df.iloc[row_idx, :][cols], treatment_schedule=treatment_schedule,
                                time=np.arange(0,len(treatment_schedule),1), dt=1, consts=consts, params=params)
            x_y = self.ode.simulate()
            ax.plot(self.ode.time, x_y[:, 0], color="b", label="x (Model)", **kwargs)
            ax.plot(self.ode.time, x_y[:, 1], color="g", label="y (Model)", **kwargs)
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)

        return ax

