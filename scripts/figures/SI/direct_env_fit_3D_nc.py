import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.lvelias import ODEModel
import arviz as az
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import pymc as pm
from scipy.optimize import least_squares
from physilearning.envs import LvEnv
import os

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule, sim_end=sim_end)

def plot_data(ax, dat, lw=2, title="Initial data"):
    ax.plot(dat.time, dat.x, color="b", lw=lw, marker="o", markersize=12, label="X (Data)")
    ax.plot(dat.time, dat.y, color="g", lw=lw, marker="+", markersize=14, label="Y (Data)")
    ax.plot(dat.time, dat.x + dat.y, color="k", lw=lw, label="Total")
    # fill between when treatment is on
    # ax.fill_between(data.time, 0, max(data.x), where=treatment_schedule == 1, facecolor='orange', alpha=0.5,
    #                 label="Treatment")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax

def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ['K', 'r_s']
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
    title="Title",
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
    out = run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule)
    return (data[["x", "y"]] - out).values.flatten()

def plot_finals():

    print(az.summary(trace))
    az.plot_trace(trace, kind="rank_bars")
    plt.suptitle(f"Trace Plot {sampler}")
    #fig, ax = plt.subplots(figsize=(12, 4))
    #plot_inference(ax, trace, title=f"Data and Inference Model Runs\n{sampler} Sampler")

    # get mean and median of distribution
    trace_df = az.summary(trace)

    # plot mean parameters
    for i, df in enumerate(data_list):

        sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
        data = pd.DataFrame(dict(
            time=df.index.values,
            x=df['Type 0'].values,
            y=df['Type 1'].values, ))

        treatment_schedule = np.array(df['Treatment'].values)

        fig, ax = plt.subplots(figsize=(12, 4))
        plot_data(ax, data)
        mean_params = {}
        for key in params_fit.keys():
            mean_params[key] = trace_df.loc[key, 'mean']
        theta = [mean_params[key] for key in params_fit.keys()]
        sol = run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule, sim_end=sim_end)
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="--", markersize=12, label="X (Mean)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="--", markersize=14, label="Y (Mean)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="--", markersize=14, label="Total (Mean)")
        ax.legend()

        median_params = {}
        for key in params_fit.keys():
            median_params[key] = trace.get('posterior').to_dataframe()[key].median()
            print(key, median_params[key])

        theta = [median_params[key] for key in params_fit.keys()]
        sol = run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule, sim_end=sim_end)
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Median)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Median)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
        ax.legend()
        ax.set_title('Median parameters')

def run_model(theta, y0, treatment, sim_end):
    # parameters distinction

    r_s = theta[0]
    #r_r = theta[1]
    #K = theta[1]
    #delta_s = theta[3]
    #t0 = theta[4]
    #k = theta[5]
    #c_r = theta[2]

    # env setup

    config_file = 'config.yaml'
    env = LvEnv.from_yaml(config_file)
    env.initial_wt = y0[0]
    env.initial_mut = y0[1]
    env.treatment_time_step = 1

    #env.death_rate_treat[0] = delta_s
    env.growth_rate[0] = r_s
    #env.growth_rate[1] = r_r
    #env.capacity = K
    #env.t0 = t0
    #env.k = k
    #env.competition[0] = c_r
    env.end_time = sim_end

    env.normalize = False
    env.reset()
    result = np.array([[env.initial_wt, env.initial_mut]])
    #result = np.append(result,[[env.state[0], env.state[1]]], axis=0)

    treat = treatment[1:]
    for i, t in enumerate(treat):
        env.step(t)
        result = np.append(result, [[env.state[0], env.state[1]]], axis=0)

    return result

if __name__ == '__main__':

    os.chdir('/home/saif/Projects/PhysiLearning')

    ############################# MTD data #############################
    # Plate3 D2 MTd
    df_mtd = pd.read_hdf('./data/3D_manuals/mtd/mtd_all.h5', key='run_1')
    df_at50 = pd.read_hdf('./data/3D_manuals/at50/at50_all.h5', key='run_1')
    df_nc = pd.read_hdf('./data/3D_manuals/nc/nc_all.h5', key='run_1')

    data_list = [df_nc]

    iteration = 1
    accuracy = 0.0
    tune_draws = 500
    final_draws = 1000
    params_fit = {'r_s': 0.323}
    sigmas = [0.005]

    while accuracy < 0.99:
        theta_fit = list(params_fit.values())
        with pm.Model() as model:
            # Shared priors
            r_s = pm.TruncatedNormal("r_s", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
                                     upper=1)
            # K = pm.TruncatedNormal("K", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=80000,
            #                           upper=500000)
            # c_r = pm.TruncatedNormal("c_r", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-2,
            #                         upper=3)

            for i, df in enumerate(data_list):

                sigma = pm.HalfNormal(f"sigma_{i}", 10)
                sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
                data = pd.DataFrame(dict(
                    time=df.index.values[:sim_end],
                    x=df['Type 0'].values[:sim_end],
                    y=df['Type 1'].values[:sim_end], ))

                treatment_schedule = np.array(df['Treatment'].values[:sim_end])

                sol = run_model(theta=[r_s], y0=[data.x[0], data.y[0]], treatment=treatment_schedule, sim_end=sim_end)
                # Ode solution function
                ode_solution = pytensor_forward_model_matrix(
                    pm.math.stack([r_s])
                )

                # Likelihood
                pm.Normal(f"Y_obs_exp{i}", mu=ode_solution, sigma=sigma, observed=data[["x", "y"]].values)

        # Variable list to give to the sample step parameter
        vars_list = list(model.values_to_rvs.keys())[:-1]

        sampler = "DEMetropolis"
        chains = 8
        draws = tune_draws
        with model:
            trace_DEM = pm.sample(tune=2 * draws, draws=draws, chains=chains, cores=16)
        trace = trace_DEM
        params_old = params_fit
        trace_df = az.summary(trace)
        params_new = {}
        for key in params_fit.keys():
            params_new[key] = trace_df.loc[key, 'mean']
        sigmas = [max(trace_df.loc[key, 'sd'], 0.001) for key in params_fit.keys()]
        params_fit = params_new

        # calculate accuracy as the difference between the old and new parameters
        accuracy_list = []
        for key in params_fit.keys():
            accuracy_list.append(1 - abs(params_old[key] - params_new[key]) / params_old[key])
        accuracy = np.min(accuracy_list)
        print("Accuracy: ", accuracy)

        print("Iteration: ", iteration)
        iteration += 1

    # final
    theta_fit = list(params_fit.values())
    with pm.Model() as model:
        # Shared priors
        r_s = pm.TruncatedNormal("r_s", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
                                 upper=1)
        # K = pm.TruncatedNormal("K", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=80000,
        #                           upper=500000)
        # c_r = pm.TruncatedNormal("c_r", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-2,
        #                         upper=3)

        for i, df in enumerate(data_list):
            sigma = pm.HalfNormal(f"sigma_{i}", 10)
            sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
            data = pd.DataFrame(dict(
                time=df.index.values[:sim_end],
                x=df['Type 0'].values[:sim_end],
                y=df['Type 1'].values[:sim_end], ))

            treatment_schedule = np.array(df['Treatment'].values[:sim_end])

            sol = run_model(theta=[r_s], y0=[data.x[0], data.y[0]], treatment=treatment_schedule, sim_end=sim_end)
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([r_s])
            )

            # Likelihood
            pm.Normal(f"Y_obs_exp{i}", mu=ode_solution, sigma=sigma, observed=data[["x", "y"]].values)

    # Variable list to give to the sample step parameter
    vars_list = list(model.values_to_rvs.keys())[:-1]

    sampler = "DEMetropolis"
    chains = 8
    draws = final_draws
    with model:
        trace_DEM = pm.sample(tune=2 * draws, draws=draws, chains=chains, cores=16)
    trace = trace_DEM

    trace.to_json('./data/SI_data/3D_12112024_nc_LV.json')
    plot_finals()
    plt.show()

    #res = run_model(theta=[0.1, 0.1, 0.1], y0=[data.x[0], data.y[0]], treatment=df['Treatment'].values)