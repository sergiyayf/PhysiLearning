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
#black background
plt.style.use('dark_background')
# set arial font and 14 font size
plt.rcParams.update({'font.size': 20})

@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix(theta):
    return ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()

def plot_data(ax, lw=0, title="Initial data"):
    data_B7 = {
        'Type 0': [6322, 7215, 8159, 8246, 6563, 5068, 5393, 6203, 6695, 8155, 10244, 12520, 13013, 0],
        'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    df = pd.DataFrame(data_B7)
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1], ))

    ax.plot(data.time, data.x, color="c", lw=lw, marker="o", markersize=12, label="Well 1")
    ax.plot(data.time, data.y, color="yellow", lw=lw, marker="o", markersize=14)

    data_C7 = {
        'Type 0': [5845, 6865, 7957, 8014, 6781, 5245, 5274, 6242, 6550, 7790, 9647, 11916, 12452, 0],
                 'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    df = pd.DataFrame(data_C7)
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1], ))

    ax.plot(data.time, data.x, color="c", lw=lw, marker="x", markersize=12, label="Well 2")
    ax.plot(data.time, data.y, color="yellow", lw=lw, marker="x", markersize=14)

    data_D7 = {
        'Type 0': [6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724, 0],
        'Type 1': [8, 9, 10, 18, 31, 59, 83, 144, 191, 269, 335, 398, 448, 0],
        'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    }
    df = pd.DataFrame(data_D7)
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1], ))

    ax.plot(data.time, data.x, color="c", lw=lw, marker="s", markersize=12, label="Well 3")
    ax.plot(data.time, data.y, color="y", lw=lw, marker="s", markersize=14)
    # fill between when treatment is on
    # ax.fill_between(data.time, 0, max(data.x), where=treatment_schedule == 1, facecolor='orange', alpha=0.5,
    #                 label="Treatment")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    # ax.set_title(title, fontsize=16)
    return ax

def plot_model_trace(ax, trace_df, row_idx, lw=1, alpha=0.2):
    cols = ['K']
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
    plt.suptitle(f"Trace Plot")
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_inference(ax, trace, title=f"Data and Inference Model Runs Sampler")

    # get mean and median of distribution
    trace_df = az.summary(trace)

    # plot mean parameters
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data(ax)
    mean_params = {}
    for key in params_fit.keys():
        mean_params[key] = trace_df.loc[key, 'mean']
    theta = [mean_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0], color="c", lw=4, ls="--", markersize=12)
    ax.plot(data.time, sol[:, 1], color="y", lw=4, ls="--", markersize=14)
    ax.plot(data.time, sol[:, 0] + sol[:, 1], color="w", lw=4, ls="--", markersize=14, label="LV model")
    ax.legend()
    # ax.set_title('Pulse')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Cell Count')
    fig.savefig('./../../../pulse_white.svg')


if __name__ == '__main__':

    # Get data Pulse Plate 1 B7
    # data_dict = {#'Type 0': [6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724, 0],
    #              #'Type 1': [8, 9, 10, 18, 31, 59, 83, 144, 191, 269, 335, 398, 448, 0],
    #              'Type 0': [5845, 6865, 7957, 8014, 6781, 5245, 5274, 6242, 6550, 7790, 9647, 11916, 12452, 0],
    #     'Type 1': [8, 9, 10, 18, 31, 59, 83, 144, 191, 269, 335, 398, 448, 0],
    #     'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    data_dict = {
        'Type 0': [6589, 7531, 8595, 8662, 7559, 6745, 6539, 7280, 7845, 8666, 9837, 11615, 11724, 0],
                 'Type 1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 'Treatment': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]}
    df = pd.DataFrame(data_dict)
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1], ))

    treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="Original treatment")
    # append treatmentd schedule with two zeros from the beginning, delete last 2
    treatment_schedule = np.array(
        [np.int32(i) for i in [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]])
    # shfit by 1 to the right
    # treatment_schedule = np.roll(treatment_schedule, -1)
    # find ends of treatment
    treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="PC raw data")

    consts_fit = {'Delta_s': 2.58, 'r_r': 0.22, 'r_s': 0.114}
    params_fit = {'K': 42100}
    sigmas = [1000]

    trace = az.from_json('./../../../data/SI_data/elias_lv_fit_02_K_B7.json')

    plot_finals()
    plt.show()

