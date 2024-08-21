import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.odemodel import ODEModel
import arviz as az

ex = {'font.size': 6,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def plot_data(ax, lw=0, title="Initial data"):
    ax.plot(data.time, data.x/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="b", lw=lw, marker="o", markersize=3, label="S (Data)")
    ax.plot(data.time, data.y/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="r", lw=lw, marker="o", markersize=3, label="R (Data)")
    ax.plot(data.time, (data.x + data.y)/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="k", lw=lw, marker="o", markersize=3, label="Total (Data)")
    # fill between when treatment is on
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, -1), treat)
    ax.fill_between(df.index, 1.2, 1.5, where=treat == 1, color='orange', label='drug',
                       lw=0)
    # ax.fill_between(data.time, 0, max(data.x), where=treatment_schedule == 1, facecolor='orange', alpha=0.5,
    #                 label="Treatment")
    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax

def plot_inference(
    ax,
    trace,
    num_samples=25,
    title="Hudson's Bay Company Data and\nInference Model Runs",
    plot_model_kwargs=dict(lw=1, alpha=0.2),
):
    trace_df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, lw=0)

def plot_finals():

    print(az.summary(trace))
    az.plot_trace(trace, kind="rank_bars")
    fig, ax = plt.subplots(1, 1, figsize=(400 / 72, 300 / 72), constrained_layout=False)
    plot_inference(ax, trace, title=f"Data and Inference Model Runs Sampler")

    # get mean and median of distribution
    trace_df = az.summary(trace)

    # plot mean parameters
    fig, ax = plt.subplots(1, 1, figsize=(400 / 72, 300 / 72), constrained_layout=True)
    plot_data(ax)
    mean_params = {}
    for key in params_fit.keys():
        mean_params[key] = trace_df.loc[key, 'mean']
    theta = [mean_params[key] for key in params_fit.keys()]
    sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    ax.plot(data.time, sol[:, 0]/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="b", lw=2, ls="--", label="S (Mean fit)")
    ax.plot(data.time, sol[:, 1]/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="r", lw=2, ls="--", label="R (Mean fit)")
    ax.plot(data.time, (sol[:, 0] + sol[:, 1])/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="k", lw=2, ls="--", markersize=14, label="Total (Mean fit)")
    ax.legend(loc='lower left')
    eq = (r"$\frac{dS}{dt} = S \cdot (r_S \cdot (1 - \frac{S + R}{K} )(1-D_S) - \delta_S) $")
    eq2 = (r"$\frac{dR}{dt} = R \cdot (r_R \cdot (1 - \frac{S + R}{K} ) - \delta_R) $")
    title = eq + "\n" + eq2 + "\n" + r"$r_S = 0.268, r_R = 0.248, K = 1.423, D_S = 5.55$" + r" fit: $D_S, K$"
    # put the equation on the top as title
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Time to progression')
    ax.set_ylabel('Normalized cell number')
    fig.savefig(
        r'/home/saif/Projects/PhysiLearning/data/si_figure_panels/capacity_at100_fit.svg',
        transparent=True)
    # plot median parameters
    #
    # median_params = {}
    # for key in params_fit.keys():
    #     median_params[key] = trace.get('posterior').to_dataframe()[key].median()
    #     print(key, median_params[key])
    #
    # theta = [median_params[key] for key in params_fit.keys()]
    # sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
    #                 params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
    # ax.plot(data.time, sol[:, 0]/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="b", lw=2, ls="-.", markersize=12, label="X (Median)")
    # ax.plot(data.time, sol[:, 1]/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="r", lw=2, ls="-.", markersize=14, label="Y (Median)")
    # ax.plot(data.time, (sol[:, 0] + sol[:, 1])/(df['Type 0'].values[0] + df['Type 1'].values[0]), color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
    # ax.legend()
    # ax.set_title('Median parameters')


if __name__ == '__main__':

    # Get data
    df = pd.read_hdf(
        '../../../data/2D_benchmarks/at100/2d_at100_all.h5', key='run_29')
    # find the index when all of the data is 0
    initial_size = df['Type 0'][0] + df['Type 1'][0]
    truncated = df[((df['Type 0'] + df['Type 1']) / initial_size > 1.33)]
    index = truncated.index[1]
    # replace df with zeros after index
    df.loc[index:, 'Type 0'] = 0
    df.loc[index:, 'Type 1'] = 0
    df.loc[index:, 'Treatment'] = 0
    sim_end = df.index[np.where(~df.any(axis=1))[0][0]]

    data = pd.DataFrame(dict(
        time=df.index.values[0:sim_end:1],
        x=df['Type 0'].values[0:sim_end:1],
        y=df['Type 1'].values[0:sim_end:1] ))

    treatment_schedule = np.array(df['Treatment'].values[0:sim_end:1])
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="Original treatment")
    # append treatmentd schedule with two zeros from the beginning, delete last 2
    treatment_schedule = np.array([np.int32(i) for i in np.array(df['Treatment'].values[0:sim_end:1])])
    # shfit by 1 to the right
    treatment_schedule = np.roll(treatment_schedule, -1)
    # find ends of treatment
    treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]

    # Plot data
    # fig, ax = plt.subplots(figsize=(12, 4))
    # plot_data(ax, title="PC raw data")

    consts_fit = {'Delta_r': 0.0, 'delta_r': 0.01, 'delta_s': 0.01,
                  'r_s': 0.268, 'c_s': 1.0, 'c_r': 1.0, 'Delta_s': 5.55, 'K': 0.714, 'r_r': 0.248}
    params_fit = {}
    sigmas = [0.001]
    sigmas = [0.004]

    trace = az.from_json('./../../../data/SI_data/capacity_2D_patient_x_at100_LV_inference_Data_1_33_threshold.json')
    #trace.to_json('./../../../data/SI_data/decoupled_no_comp_2D_patient_x_at100_LV_inference_Data_1_33_threshold.json')
    plot_finals()
    plt.show()

