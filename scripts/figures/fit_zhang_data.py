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
import os

@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def pytensor_forward_model_matrix(theta):
    return np.sum(ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [x0, y0],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate(), axis=1)[time]

def plot_data(ax, dat, lw=2, title="Initial data"):
    #ax.plot(dat.time, dat.tot, color="b", lw=lw, marker="o", markersize=12, label="X (Data)")
    #ax.plot(dat.time, dat.y, color="g", lw=lw, marker="+", markersize=14, label="Y (Data)")
    ax.plot(dat.time, dat.tot, color="k", lw=lw, label="Total")
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

    sim_end = 537 - 121 + 1
    treatment_schedule = np.ones(sim_end)
    data = pd.DataFrame(dict(
        time=time,
        tot=tot))
    ini_tot = 6552.52 + 166.29
    x0 = 6552.52 / ini_tot
    y0 = 166.29 / ini_tot

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, data)
    mean_params = {}
    for key in params_fit.keys():
        mean_params[key] = trace_df.loc[key, 'mean']
    theta = [mean_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [x0, y0],
                params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

    ax.plot(data.time, sol[time, 0], color="r", lw=2, ls="--", markersize=12, label="X (Mean)")
    ax.plot(data.time, sol[time, 1], color="g", lw=2, ls="--", markersize=14, label="Y (Mean)")
    ax.plot(data.time, sol[time, 0] + sol[time, 1], color="k", lw=2, ls="--", markersize=14, label="Total (Mean)")
    ax.legend()

    median_params = {}
    for key in params_fit.keys():
        median_params[key] = trace.get('posterior').to_dataframe()[key].median()
        print(key, median_params[key])

    theta = [median_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [x0, y0],
                params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[time, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Median)")
    ax.plot(data.time, sol[time, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Median)")
    ax.plot(data.time, sol[time, 0] + sol[time, 1], color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
    ax.legend()
    ax.set_title('Median parameters')

if __name__ == '__main__':

    os.chdir('/')

    ############################# MTD data #############################
    time = np.array([121, 156, 206, 248, 297, 339, 352, 382, 423, 466, 494, 537]) - 121
    tot = np.array([4.4, 0.9, 0.18, 0.12, 0.1, 0.25, 0.32, 0.7, 1.35, 2.2, 2.3, 3.35]) / 4.4
    treat = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    iteration = 1
    accuracy = 0.0
    tune_draws = 100
    final_draws = 100
    consts_fit = {'r_s': 0.0045,  'delta_s': 0.01, 'delta_r': 0.01, 'c_s': 4.82, 'c_r': 1.0, 'K': 18000, 'Delta_r': 0.0, 'Delta_s': 0.01}
    params_fit = {'r_r': 0.0134}
    sigmas = [0.005]
    while accuracy < 0.099:
        theta_fit = list(params_fit.values())
        with pm.Model() as model:
            # Shared priors

            r_r = pm.TruncatedNormal("r_r", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
                                        upper=1)

            # Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=1.e-2,
            #                              upper=1000)

            sigma = pm.HalfNormal(f"sigma", 10)
            sim_end = 537-121+1
            treatment_schedule = np.ones(sim_end)
            data = pd.DataFrame(dict(
                time=time,
                tot=tot ))
            ini_tot = 6552.52+166.29
            x0 = 6552.52/ini_tot
            y0 = 166.29/ini_tot
            sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0 = [x0, y0],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
            # Ode solution function
            sol = np.sum(sol, axis=1)[time]
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([r_r])
            )

            # Likelihood
            pm.Normal(f"Y_obs_exp", mu=ode_solution, sigma=sigma, observed=data[["tot"]].values)

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
    # with pm.Model() as model:
    #     # Shared priors
    #     r_r = pm.TruncatedNormal("r_r", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-2,
    #                              upper=1)
    #
    #     # Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=1.e-2,
    #     #                              upper=1000)
    #
    #     sigma = pm.HalfNormal(f"sigma", 10)
    #     sim_end = 537 - 121 + 1
    #     treatment_schedule = np.ones(sim_end)
    #     data = pd.DataFrame(dict(
    #         time=time,
    #         tot=tot))
    #     ini_tot = 6552.52 + 166.29
    #     x0 = 6552.52 / ini_tot
    #     y0 = 166.29 / ini_tot
    #     sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[x0, y0],
    #                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    #     # Ode solution function
    #     sol = np.sum(sol, axis=1)[time]
    #     ode_solution = pytensor_forward_model_matrix(
    #         pm.math.stack([r_r])
    #     )
    #
    #     # Likelihood
    #     pm.Normal(f"Y_obs_exp", mu=ode_solution, sigma=sigma, observed=data[["tot"]].values)

    # Variable list to give to the sample step parameter
    # vars_list = list(model.values_to_rvs.keys())[:-1]
    #
    # sampler = "DEMetropolis"
    # chains = 8
    # draws = final_draws
    # with model:
    #     trace_DEM = pm.sample(tune=2 * draws, draws=draws, chains=chains, cores=16)
    # trace = trace_DEM

    #trace.to_json('./data/SI_data/2D_29112024_mtd_LV.json')
    plot_finals()
    plt.show()

    #res = run_model(theta=[0.1, 0.1, 0.1], y0=[data.x[0], data.y[0]], treatment=df['Treatment'].values)