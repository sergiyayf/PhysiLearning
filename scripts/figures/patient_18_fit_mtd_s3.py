import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.odemodel import ODEModel
import arviz as az
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import pymc as pm
from scipy.optimize import least_squares

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

def plot_data(ax, lw=2, title="Initial data"):
    ax.plot(data.time, data.x, color="b", lw=lw, marker="o", markersize=12, label="X (Data)")
    ax.plot(data.time, data.y, color="g", lw=lw, marker="+", markersize=14, label="Y (Data)")
    ax.plot(data.time, data.x + data.y, color="k", lw=lw, label="Total")
    # fill between when treatment is on
    ax.fill_between(data.time, 0, max(data.x), where=treatment_schedule == 1, facecolor='orange', alpha=0.5,
                    label="Treatment")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax

def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ['r_r', 'delta_r', 'Delta_s']
    row = trace_df.iloc[row_idx, :][cols].values

    theta = row
    x_y = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                     params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                        params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).plot_model(ax, solution=x_y, lw=lw, alpha=alpha)

def plot_inference(
    ax,
    trace,
    num_samples=25,
    title="Hudson's Bay Company Data and\nInference Model Runs",
    plot_model_kwargs=dict(lw=1, alpha=0.2),
):
    trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, lw=0)
    for row_idx in range(num_samples):
        plot_model_trace(ax, trace_df, row_idx, **plot_model_kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)

def ode_model_resid(theta):
    out = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

    return (data[["x", "y"]] - out).values.flatten()

def plot_finals():

    print(az.summary(trace))
    az.plot_trace(trace, kind="rank_bars")
    plt.suptitle(f"Trace Plot {sampler}")
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler")

    # get mean and median of distribution
    trace_df = az.summary(trace)

    # plot mean parameters
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax)
    mean_params = {}
    for key in params_fit.keys():
        mean_params[key] = trace_df.loc[key, 'mean']
    theta = [mean_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Mean)")
    ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Mean)")
    ax.legend()
    ax.set_title('Mean parameters')

    # plot median parameters

    median_params = {}
    for key in params_fit.keys():
        median_params[key] = trace.get('posterior').to_dataframe()[key].median()
    theta = [median_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Median)")
    ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Median)")
    ax.legend()
    ax.set_title('Median parameters')


if __name__ == '__main__':

    # Get data
    # df = pd.read_csv(
    #     './../Evaluations/older_evals/0_AT_fixedAT_60onPC.csv',
    #     index_col=[0])
    #
    # data = pd.DataFrame(dict(
    #         time=df.index.values[0:230:1],
    #         x=df['Type 0'].values[0:230:1],
    #         y=df['Type 1'].values[0:230:1],))
    #
    # treatment_schedule = [np.int32(i) for i in np.array(df['Treatment'].values[0:230:1])]

    # Get data
    df = pd.read_csv(
        './../../Evaluations/0_PcEnvEvalpatient_18_mtd.csv',
        index_col=[0])
    # find the index when all of the data is 0
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1], ))

    treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="Original treatment")
    # append treatmentd schedule with two zeros from the beginning, delete last 2
    treatment_schedule = np.array([np.int32(i) for i in np.array(df['Treatment'].values[0:sim_end:1])])
    # find ends of treatment
    treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]
    # replace ends of treatment with 1
    # treatment_schedule[treatment_ends + 1] = 1

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="PC raw data")

    consts_fit = {'Delta_r': 0.0, 'K': 5000, 'c_s': 1.5, 'c_r': 1,
              'delta_s': 0.0001, 'r_s': 0.24}
    params_fit = {'r_r': 0.4, 'delta_r': 0.1, 'Delta_s': 0.35}

    theta_fit = list(params_fit.values())

    sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[data.x[0], data.y[0]],
                   params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="--", markersize=12, label="X (Initial guess)")
    ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="--", markersize=14, label="Y (Initial guess)")

    initial_conditions = least_squares(ode_model_resid, x0=list(params_fit.values()), bounds=(1e-4,1))
    params_fit = {'r_r': initial_conditions.x[0], 'delta_r': initial_conditions.x[1], 'Delta_s': initial_conditions.x[2]}
    theta_fit = list(params_fit.values())

    sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Least squares)")
    ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Least squares)")
    ax.legend()

    with pm.Model() as model:
        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        # r_r = pm.Uniform("r_r", lower=0, upper=1, initval=theta_fit[0])
        # r_s = pm.Uniform("r_s", lower=0, upper=1, initval=theta_fit[1])
        # r_r = pm.TruncatedNormal("r_r", mu=theta_fit[0], sigma=0.1*theta_fit[0], lower=0, initval=theta_fit[0])
        r_r = pm.TruncatedNormal("r_r", mu=theta_fit[0], sigma=0.1*theta_fit[0], lower=0, initval=theta_fit[0])
        delta_r = pm.TruncatedNormal("delta_r", mu=theta_fit[1], sigma=0.1*theta_fit[1], lower=0, initval=theta_fit[1])
        Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[2], sigma=0.1*theta_fit[2], lower=0, initval=theta_fit[2])

        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([r_r, delta_r, Delta_s])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data[["x", "y"]].values)

    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    sampler = "DEMetropolis"
    chains = 8
    draws = 5
    with model:
        trace_DEM = pm.sample(step=[pm.DEMetropolis(vars_list)], tune=2 * draws, draws=draws, chains=chains)
    trace = trace_DEM

    plot_finals()
    plt.show()

