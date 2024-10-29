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
from physilearning.envs import ArrEnv
import os

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule)

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
    for i, data_dict in enumerate(data_list):
        df = pd.DataFrame(data_dict)

        data = pd.DataFrame(dict(
            time=df.index.values[1:10],
            x=df['Type 0'].values[1:10],
            y=df['Type 1'].values[1:10], ))

        treatment_schedule = np.array(df['Treatment'].values[1:10])

        fig, ax = plt.subplots(figsize=(12, 4))
        plot_data(ax, data)
        mean_params = {}
        for key in params_fit.keys():
            mean_params[key] = trace_df.loc[key, 'mean']
        theta = [mean_params[key] for key in params_fit.keys()]
        sol = run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule)
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="--", markersize=12, label="X (Mean)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="--", markersize=14, label="Y (Mean)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="--", markersize=14, label="Total (Mean)")
        ax.legend()

        median_params = {}
        for key in params_fit.keys():
            median_params[key] = trace.get('posterior').to_dataframe()[key].median()
            print(key, median_params[key])

        theta = [median_params[key] for key in params_fit.keys()]
        sol = run_model(theta=theta, y0=[data.x[0], data.y[0]], treatment=treatment_schedule)
        ax.plot(data.time, sol[:, 0], color="r", lw=2, ls="-.", markersize=12, label="X (Median)")
        ax.plot(data.time, sol[:, 1], color="g", lw=2, ls="-.", markersize=14, label="Y (Median)")
        ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
        ax.legend()
        ax.set_title('Median parameters')

def run_model(theta, y0, treatment):
    # parameters distinction

    rec_rate = theta[0]

    # env setup

    config_file = 'config.yaml'
    env = ArrEnv.from_yaml(config_file)
    env.initial_wt = y0[0]
    env.initial_mut = y0[1]
    env.treatment_time_step = 2

    env.recover_rate = rec_rate

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
    mtd1 = {'Type 0': [6980, 6883, 7888, 8986, 7665, 6079, 4157, 3103, 1823, 1426, 911, 803, 493, 426],
          'Type 1': [0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
          'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # Plate3 D4 mtd
    mtd2 = {'Type 0': [7071, 7308, 8537, 9314, 8011, 6246, 4279, 3008, 1932, 1534, 1127, 954, 642, 534],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # Plate3 D7 mtd
    mtd3 = {'Type 0': [7648, 7833, 9194, 9766, 8690, 7038, 4444, 3178, 1895, 1430,  950,  766,  566,  491],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    ############################# eAT100 data #############################

    # Plate1 D2
    treat1_1 = {'Type 0': [7245, 7074, 8394, 9172, 8018, 6935, 6292, 6961, 7593, 8577],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}
    # Plate1 D5
    treat1_2 = {'Type 0': [7714, 7624, 8820, 9681, 8354, 7423, 6401, 7223, 7965, 9694],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}
    # Plate1 D7
    treat1_3 = {'Type 0': [7764, 7750, 8552, 9566, 8483, 7539, 6788, 7637, 8533, 9700],
                'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Treatment': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]}

    ############################# AT100 data #############################
    # Plate1 E4
    treat2_1 = {'Type 0': [7831, 7793, 9121, 9667, 8485, 6922, 4689, 3607, 2694, 3117, 3797, 5218, 6657, 8613],
                'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Treatment': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate1 E5
    treat2_2 = {'Type 0': [7908, 7771, 9121, 9786, 8530, 7032, 4905, 3701, 2767, 2984, 3602, 4751, 6232, 7766],
                'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Treatment': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate1 E6
    treat2_3 = {'Type 0': [8223, 8030, 9134, 9993, 8781, 7203, 4885, 3795, 2856, 3254, 3644, 5147, 6572, 7567],
                'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Treatment': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}

    ############################# AT50 data #############################
    # C2, C3, C5
    # Plate1 C2
    treat3_1 = {'Type 0': [7071, 7267, 7954, 8794, 7637, 6239, 4710, 3614, 2413, 2024, 1910, 2417, 3493, 4924, 6307, 7867],
                'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate1 C3
    treat3_2 = {
        'Type 0': [7361, 7425, 8520, 9250, 8107, 6612, 4695, 3501, 2256, 1980, 1674, 2253, 3005, 4444, 5731, 7227],
        'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate1 C5
    treat3_3 = {
        'Type 0': [7453, 7434, 8628, 9328, 7942, 6460, 4543, 3359, 2185, 1693, 1496, 1910, 2556, 4162, 5486, 6752],
        'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}

    ############################# nc data #############################
    #E3, E6, E7
    # Plate3 E3
    nc_1 = {'Type 0': [7022, 7368, 8799, 10946, 12655, 15305, 16515, 19284, 20086, 24395, 27738],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate3 E6
    nc_2 = {'Type 0': [7687, 7859, 9194, 11732, 13399, 16037, 17012, 20622, 19677, 23456, 26791],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # Plate3 E7
    nc_3 = {'Type 0': [7620, 7865, 9520, 11775, 13537, 16468, 17914, 20766, 22366, 24169, 27038],
            'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Treatment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    #data_list = [mtd1, mtd2, mtd3, treat1_1, treat1_2, treat1_3, treat2_1, treat2_2, treat2_3, treat3_1, treat3_2, treat3_3, nc_1, nc_2, nc_3]
    data_list = [treat1_1]

    iteration = 1
    accuracy = 0.0
    tune_draws = 100
    final_draws = 100
    consts_fit = {'r_r': 1.25}
    params_fit = {'rec_r': 0.25}
    sigmas = [0.1]

    while accuracy < 0.9:
        theta_fit = list(params_fit.values())
        with pm.Model() as model:
            # Shared priors

            rec_r = pm.TruncatedNormal("rec_r", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-3,
                                        upper=1)


            for i, data_dict in enumerate(data_list):
                df = pd.DataFrame(data_dict)
                sigma = pm.HalfNormal(f"sigma_{i}", 10)

                data = pd.DataFrame(dict(
                    time=df.index.values[1:10],
                    x=df['Type 0'].values[1:10],
                    y=df['Type 1'].values[1:10], ))

                treatment_schedule = np.array(df['Treatment'].values[1:10])

                sol = run_model(theta=[rec_r], y0=[data.x[0], data.y[0]], treatment=treatment_schedule)
                # Ode solution function
                ode_solution = pytensor_forward_model_matrix(
                    pm.math.stack([rec_r])
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

        rec_r = pm.TruncatedNormal("rec_r", mu=theta_fit[0], sigma=sigmas[0], initval=theta_fit[0], lower=1.e-3,
                                   upper=1)

        for i, data_dict in enumerate(data_list):
            df = pd.DataFrame(data_dict)
            sigma = pm.HalfNormal(f"sigma_{i}", 10)

            data = pd.DataFrame(dict(
                time=df.index.values[1:10],
                x=df['Type 0'].values[1:10],
                y=df['Type 1'].values[1:10], ))

            treatment_schedule = np.array(df['Treatment'].values[1:10])

            sol = run_model(theta=[rec_r], y0=[data.x[0], data.y[0]], treatment=treatment_schedule)
            # Ode solution function
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([rec_r])
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

    #trace.to_json('./data/SI_data/trace_arr_env_pulse_rec.json')
    plot_finals()
    plt.show()

    #res = run_model(theta=[0.1, 0.1, 0.1], y0=[data.x[0], data.y[0]], treatment=df['Treatment'].values)