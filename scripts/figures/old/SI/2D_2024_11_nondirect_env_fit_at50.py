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
from physilearning.envs import LvEnv
import os

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

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

def plot_finals():

    print(az.summary(trace))
    az.plot_trace(trace, kind="rank_bars")
    plt.suptitle(f"Trace Plot {sampler}")
    trace_df = az.summary(trace)

    # plot mean parameters
    for i, df in enumerate(data_list):

        sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
        data = pd.DataFrame(dict(
            time=df.index.values[0:sim_end],
            x=df['Type 0'].values[0:sim_end],
            y=df['Type 1'].values[0:sim_end], ))

        treatment_schedule = np.array(
            [np.int32(i) for i in
             df['Treatment'].values[1:sim_end+1]])

        fig, ax = plt.subplots(figsize=(12, 4))
        plot_data(ax, data)
        mean_params = {}
        for key in params_fit.keys():
            mean_params[key] = trace_df.loc[key, 'mean']
        theta = [mean_params[key] for key in params_fit.keys()]
        sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="--", markersize=12, label="X (Mean)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="--", markersize=14, label="Y (Mean)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="--", markersize=14, label="Total (Mean)")
        ax.legend()

        median_params = {}
        for key in params_fit.keys():
            median_params[key] = trace.get('posterior').to_dataframe()[key].median()
            print(key, median_params[key])

        theta = [median_params[key] for key in params_fit.keys()]
        sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Median)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Median)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
        ax.legend()
        ax.set_title('Median parameters')

if __name__ == '__main__':

    os.chdir('/')

    ############################# MTD data #############################
    # Plate3 D2 MTd
    df_mtd = pd.read_hdf('./data/3D_manuals/mtd/mtd_all.h5', key='run_1')
    df_at50 = pd.read_hdf('./data/29112024_2d_manuals/at50_demo/Evaluations/PcEnvEval_job_1388539620241129_2D_manuals_at50_demo.h5', key='run_0')


    data_list = [df_at50]

    iteration = 1
    accuracy = 0.0
    tune_draws = 1000
    final_draws = 1000
    consts_fit = {'r_r': 0.134, 'delta_s': 0.01, 'delta_r': 0.01, 'c_r': 1.0, 'K': 18000,
                  'Delta_r': 0.0}
    params_fit = {'Delta_s': 330, 'c_s': 4.78, 'r_s': 0.040} # for classic c_r,=2.5, Delta_s = 1.57
    sigmas = [10, 0.05, 0.01]

    while accuracy < 0.99:
        theta_fit = list(params_fit.values())
        with pm.Model() as model:
            # Shared priors
            Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
                                     upper=1000)
            # K = pm.TruncatedNormal("K", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=80000,
            #                           upper=200000)
            c_s = pm.TruncatedNormal("c_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=1.e-2,
                                    upper=8)
            r_s = pm.TruncatedNormal("r_s", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-2,
                                    upper=1)


            for i, df in enumerate(data_list):

                sigma = pm.HalfNormal(f"sigma_{i}", 10)
                sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
                data = pd.DataFrame(dict(
                    time=df.index.values[0:sim_end],
                    x=df['Type 0'].values[0:sim_end],
                    y=df['Type 1'].values[0:sim_end], ))

                treatment_schedule = np.array(
                    [np.int32(i) for i in
                     df['Treatment'].values[1:sim_end + 1]])

                sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[data.x[0], data.y[0]],
                               params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
                # Ode solution function
                ode_solution = pytensor_forward_model_matrix(
                    pm.math.stack([Delta_s, c_s, r_s])
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
        Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
                                     upper=1000)
        c_s = pm.TruncatedNormal("c_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=1.e-2,
                                 upper=8)
        r_s = pm.TruncatedNormal("r_s", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-2,
                                 upper=1)

        for i, df in enumerate(data_list):
            sigma = pm.HalfNormal(f"sigma_{i}", 10)
            sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
            data = pd.DataFrame(dict(
                time=df.index.values[0:sim_end],
                x=df['Type 0'].values[0:sim_end],
                y=df['Type 1'].values[0:sim_end], ))

            treatment_schedule = np.array(
                [np.int32(i) for i in
                 df['Treatment'].values[1:sim_end + 1]])

            sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[data.x[0], data.y[0]],
                           params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([Delta_s, c_s, r_s])
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

    trace.to_json('./data/SI_data/2D_29112024_at50_LV.json')
    plot_finals()
    plt.show()

    #res = run_model(theta=[0.1, 0.1, 0.1], y0=[data.x[0], data.y[0]], treatment=df['Treatment'].values)