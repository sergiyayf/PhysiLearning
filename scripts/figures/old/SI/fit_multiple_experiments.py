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

def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ['K', 'Delta_s', 'r_s']
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
    out = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

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
    for i, data_dict in enumerate(data_list):
        df = pd.DataFrame(data_dict)

        sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

        data = pd.DataFrame(dict(
            time=df.index.values[0:sim_end:1],
            x=df['Type 0'].values[0:sim_end:1],
            y=df['Type 1'].values[0:sim_end:1], ))
        treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
        # transform treatment schedule
        if treatment_schedule[5] == 0:
            # append treatmentd schedul e with two zeros from the beginning, delete last 2
            treatment_schedule = np.array(
                [np.int32(i) for i in
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        else:
            treatment_schedule = np.array(
                [np.int32(i) for i in
                 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

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

    # Get data Pulse Plate 1 B7
    B7 = {'Type 0': [6322, 7215, 8159, 8246, 6563, 5068, 5393, 6203, 6695, 8155, 10244, 12520, 13013, 0],
          'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    C7 = {'Type 0': [5845, 6865, 7957, 8014, 6781, 5245, 5274, 6242, 6550, 7790, 9647, 11916, 12452, 0],
          'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    D7 = {'Type 0': [6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724, 0],
          'Type 1': [8, 9, 10, 18, 31, 59, 83, 144, 191, 269, 335, 398, 448, 0],
          'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    E7 = {'Type 0': [7008, 7883, 8928, 8922, 7559, 5748, 4115, 2870, 1596, 1083, 651, 455, 343, 0],
          'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}

    data_list = [E7, D7, C7, B7]


    iteration = 1
    accuracy = 0.0
    tune_draws = 1000
    final_draws = 10000
    consts_fit = {'r_r': 1.25}
    params_fit = {'K': 48300, 'Delta_s': 2.076, 'r_s': 0.096}
    sigmas = [1210, 0.005, 0.001]

    while accuracy < 0.99:
        theta_fit = list(params_fit.values())
        with pm.Model() as model:
            # Shared priors


            K = pm.TruncatedNormal("K", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=10000, upper=120000)
            Delta_s = pm.TruncatedNormal("Delta_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1], lower=1.e-3, upper=10)
            #r_r = pm.TruncatedNormal("r_r", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-3, upper=1)
            r_s = pm.TruncatedNormal("r_s", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-3, upper=1)

            for i, data_dict in enumerate(data_list):
                df = pd.DataFrame(data_dict)
                sigma = pm.HalfNormal(f"sigma_{i}", 10)

                sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

                data = pd.DataFrame(dict(
                    time=df.index.values[0:sim_end:1],
                    x=df['Type 0'].values[0:sim_end:1],
                    y=df['Type 1'].values[0:sim_end:1], ))

                treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
                # transform treatment schedule
                if treatment_schedule[5] == 0:
                    # append treatmentd schedul e with two zeros from the beginning, delete last 2
                    treatment_schedule = np.array(
                        [np.int32(i) for i in
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                else:
                    treatment_schedule = np.array(
                        [np.int32(i) for i in
                         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

                # shfit by 1 to the right
                # treatment_schedule = np.roll(treatment_schedule, -1)
                # find ends of treatment
                treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]

                sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[data.x[0], data.y[0]],
                               params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
                # Ode solution function
                ode_solution = pytensor_forward_model_matrix(
                    pm.math.stack([K, Delta_s, r_s])
                )

                # Likelihood
                pm.Normal(f"Y_obs_exp{i}", mu=ode_solution, sigma=sigma, observed=data[["x", "y"]].values)

        # Variable list to give to the sample step parameter
        vars_list = list(model.values_to_rvs.keys())[:-1]

        sampler = "DEMetropolis"
        chains = 8
        draws = tune_draws
        with model:
            trace_DEM = pm.sample( tune=2 * draws, draws=draws, chains=chains, cores=16)
        trace = trace_DEM
        params_old = params_fit
        trace_df = az.summary(trace)
        params_new = {}
        for key in params_fit.keys():
            params_new[key] = trace_df.loc[key, 'mean']
        sigmas = [max(trace_df.loc[key, 'sd'],0.001) for key in params_fit.keys()]
        params_fit = params_new

        # calculate accuracy as the difference between the old and new parameters
        accuracy_list = []
        for key in params_fit.keys():
            accuracy_list.append(1-abs(params_old[key] - params_new[key])/params_old[key])
        accuracy = np.min(accuracy_list)
        print("Accuracy: ", accuracy)

        print("Iteration: ", iteration)
        iteration+=1

    # final
    theta_fit = list(params_fit.values())
    with pm.Model() as model:
        # Shared priors

        K = pm.TruncatedNormal("K", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=10000, upper=120000)
        Delta_s = pm.Normal("Delta_s", mu=theta_fit[1], sigma=sigmas[1], initval=theta_fit[1])
        # r_r = pm.TruncatedNormal("r_r", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-3, upper=1)
        r_s = pm.TruncatedNormal("r_s", mu=theta_fit[2], sigma=sigmas[2], initval=theta_fit[2], lower=1.e-3, upper=1)

        for i, data_dict in enumerate(data_list):
            df = pd.DataFrame(data_dict)
            sigma = pm.HalfNormal(f"sigma_{i}", 10)

            sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

            data = pd.DataFrame(dict(
                time=df.index.values[0:sim_end:1],
                x=df['Type 0'].values[0:sim_end:1],
                y=df['Type 1'].values[0:sim_end:1], ))

            treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
            # transform treatment schedule
            if treatment_schedule[5] == 0:
                # append treatmentd schedul e with two zeros from the beginning, delete last 2
                treatment_schedule = np.array(
                    [np.int32(i) for i in
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            else:
                treatment_schedule = np.array(
                    [np.int32(i) for i in
                     [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

            # shfit by 1 to the right
            # treatment_schedule = np.roll(treatment_schedule, -1)
            # find ends of treatment
            treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]

            sol = ODEModel(theta=theta_fit, treatment_schedule=treatment_schedule, y0=[data.x[0], data.y[0]],
                           params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([K, Delta_s, r_s])
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

    trace.to_json('./../../../data/SI_data/multi_fit_experiment_with_turnover.json')
    plot_finals()
    plt.show()

