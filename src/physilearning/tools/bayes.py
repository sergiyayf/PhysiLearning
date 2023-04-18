import pymc as pm
import arviz as az
import numpy as np
from physilearning.tools.odemodel import ODEModel
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

class ODEBayesianFitter():
    def __init__(self, ode=ODEModel(), data=None):
        """Initialize the ODEBayesianFitter class.

        Parameters
        ----------
        ode : ODEModel
            The ODE model to fit.
        data : array-like
            The data to fit the model to.

        Attributes
        ----------
        model : pymc.Model
            The PyMC4 model.
        data : array-like
            The data to fit the model to.
        ode : ODEModel
            The ODE model to fit.
        trace : pymc.backends.base.MultiTrace
            The trace of the MCMC sampling.

        Methods
        -------
        set_priors(prior_dist="normal", **kwargs)
            Define the priors for the model parameters.
        set_likelihood(likelihood_dist="normal", **kwargs)
            Define the likelihood for the model.
        fit(n_samples=1000, tune=1000, cores=1, chains=1, **kwargs)
            Fit the model to the data.
        plot_trace()
            Plot the trace of the MCMC sampling.

        """

        self.model = pm.Model()
        self.data= data
        self.ode = ode
        self.trace = None


    def set_priors(self, prior_dist="normal"):
        """Define the priors for the model parameters.

        Parameters
        ----------
        prior_dist : str
            The distribution to use for the priors.

        Returns
        -------
        dict
            The priors for the model parameters.
        """
        if prior_dist == "normal":
            with self.model:
                alpha = pm.TruncatedNormal("alpha", mu=self.ode.params[0], sigma=self.ode.params[0], lower=0, initval=self.ode.params[0])
                beta = pm.TruncatedNormal("beta", mu=self.ode.params[1], sigma=self.ode.params[1], lower=0, initval=self.ode.params[1])
                gamma = pm.TruncatedNormal("gamma", mu=self.ode.params[2], sigma=self.ode.params[2], lower=0, initval=self.ode.params[2])
                delta = pm.TruncatedNormal("delta", mu=self.ode.params[3], sigma=self.ode.params[3], lower=0, initval=self.ode.params[3])
                xt0 = pm.TruncatedNormal("xt0", mu=self.ode.y0[0], sigma=self.ode.y0[0], lower=0, initval=self.ode.y0[0])
                yt0 = pm.TruncatedNormal("yt0", mu=self.ode.y0[1], sigma=self.ode.y0[1], lower=0, initval=self.ode.y0[1])
                sigma = pm.TruncatedNormal("sigma", mu=10, sigma=10, lower=0)

        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta, "xt0": xt0, "yt0": yt0, "sigma": sigma}

    @staticmethod
    @as_op(itypes=[pt.dvector, pt.dvector, pt.ivector, pt.dvector], otypes=[pt.dmatrix])
    def pytensor_matrix_solve(y0,times,treatment_schedule,theta):
        """Define the ODE with pytensor inputs and outputs.

        Parameters
        ----------
        y0 : pt.dvector
            The initial conditions for the ODE.
        times : pt.dvector
            The times to solve the ODE at.
        treatment_schedule : pt.ivector
            The treatment schedule for the ODE.
        theta : pt.dvector
            The parameters for the ODE.

        Returns
        -------
        solution: pt.dmatrix
            The solution to the ODE.
        """
        return ODEModel(y0=y0, params=theta, time=times, dt=1, treatment_schedule=treatment_schedule).simulate()

    def likelihood(self, priors):
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

            times = pm.math.stack([np.float64(t) for t in self.data["time"].values])
            theta = pm.math.stack([priors["alpha"], priors["beta"], priors["gamma"], priors["delta"]])
            y0 = pm.math.stack([priors["xt0"], priors["yt0"]])
            treatment_schedule = pm.math.stack(self.ode.treatment_schedule)
            ode_solution = self.pytensor_matrix_solve(y0,times,treatment_schedule,theta)
            likelihood = pm.Normal("likelihood", mu=ode_solution, sigma=sigma, observed=self.data[["x", "y"]].values)

        return likelihood

    def sample(self, sampler="DEMetropolis", chains=8, draws=3000, cores=1):
        """
        Fit the Lotka-Volterra model to the data using a Bayesian approach.

        Returns
        -------
        trace : pymc.backends.base.MultiTrace
            The trace of the MCMC sampling.
        """

        priors = self.set_priors()
        self.likelihood(priors=priors)
        vars_list = list(self.model.values_to_rvs.keys())[:-1]
        print(vars_list)
        sampler = sampler
        chains = chains
        draws = draws
        with self.model:
            trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], draws=draws, chains=chains, cores=cores)
        trace = trace_DEM
        self.trace = trace
        return trace

    def plot_inference_trace(self, ax=None, num_samples=25, title="DEM inference", **kwargs):
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
        cols = ["alpha", "beta", "gamma", "delta", "xt0", "yt0"]
        trace_df = az.extract(self.trace, num_samples=num_samples).to_dataframe()
        #self.time = np.arange(1900, 1921, 0.01)
        for row_idx in range(num_samples):
            theta = trace_df.iloc[row_idx, :][cols]
            self.ode.params = theta[:-2]
            self.ode.y0 = theta[-2:]
            x_y = self.ode.simulate()
            ax.plot(self.ode.time, x_y[:, 0], color="b", label="x (Model)", **kwargs)
            ax.plot(self.ode.time, x_y[:, 1], color="g", label="y (Model)", **kwargs)
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)

        return ax


# do fitting
if __name__ == "__main__":
    #treatment_schedule = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])

    treatment_schedule = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    model = ODEModel(tmax=22, treatment_schedule=treatment_schedule, dt=1)

    time = model.time
    solution = model.solve(model.time)
    sol2 = model.simulate()

    model.plot_model(solution=solution)
    model.plot_model(solution=sol2)
    fig, ax = plt.subplots()
    model.plot_model(ax=ax, solution=sol2)

    noise = np.random.normal(0, 0.001, size=solution.shape)
    sol2 += noise
    ax.scatter(time, sol2[:, 0], label='x')
    ax.scatter(time, sol2[:, 1], label='y')

    treatment_schedule = [np.int32(i) for i in treatment_schedule]
    ode_model = ODEModel(tmax=22, treatment_schedule=treatment_schedule, dt=1)
    x = sol2[:, 0]
    y = sol2[:, 1]
    data = pd.DataFrame(dict(
        time=time,
        x=x,
        y=y))
    bayes_fitter = ODEBayesianFitter(ode_model, data)
    #likelihood = bayes_fitter.likelihood(bayes_fitter.set_priors())
    trace = bayes_fitter.sample(draws=10000, chains=8)
    print(az.summary(trace))
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot_inference(ax, trace, num_samples=25)
    bayes_fitter.plot_inference_trace(ax=ax, num_samples=25, alpha=0.2)
    ax.plot(data.time, data.x, color="b", lw=2, marker="o", markersize=12, label="x")
    ax.plot(data.time, data.y, color="g", lw=2, marker="+", markersize=14, label="y")
