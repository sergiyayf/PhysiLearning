import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.odemodel import ODEModel
import arviz as az
import os

ex = {'font.size': 14,
          'font.weight': 'normal',
          'font.family': 'sans-serif'}
plt.rcParams.update(ex)
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def plot_data(ax, dat, lw=2, title="Initial data"):
    color_sus = '#5E82B8'
    ax.plot(dat.time, dat.x, color=color_sus, lw=lw, marker="o", markersize=12, label="X (Data)")
    color_res = '#EBAA42'
    ax.plot(dat.time, dat.y, color=color_res, lw=lw, marker="+", markersize=14, label="Y (Data)")
    # ax.plot(dat.time, dat.x + dat.y, color="k", lw=lw, label="Total")
    treatment_color = '#A8DADC'
    # fill between when treatment is on
    ax.fill_between(dat.time, 0, 1.3, where=treatment_schedule == 1, facecolor=treatment_color, alpha=0.5,
                    label="Treatment")
    # ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    # ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Relative tumor size')
    return ax

def plot_finals():

    print(az.summary(trace))
    az.plot_trace(trace, kind="rank_bars")
    plt.suptitle(f"Trace Plot {sampler}")
    trace_df = az.summary(trace)

    # plot mean parameters
    for i, df in enumerate(data_list):

        sim_end = df.index[np.where(~df.any(axis=1))[0][0]]
        ini = df['Type 0'][0] + df['Type 1'][0]
        data = pd.DataFrame(dict(
            time=df.index.values[0:sim_end]/4,
            x=df['Type 0'].values[0:sim_end]/ini,
            y=df['Type 1'].values[0:sim_end]/ini, ))

        treatment_schedule = np.array(
            [np.int32(i) for i in
             df['Treatment'].values[1:sim_end+1]])

        fig, ax = plt.subplots(figsize=(12, 2.5))
        plot_data(ax, data)

        median_params = {}
        for key in params_fit.keys():
            median_params[key] = trace.get('posterior').to_dataframe()[key].median()
            print(key, median_params[key])

        consts_fit['K'] = consts_fit['K']/ini
        consts_fit['Delta_s'] = consts_fit['Delta_s']/ini
        theta = [median_params[key] for key in params_fit.keys()]
        sol = ODEModel(theta=theta, treatment_schedule=treatment_schedule, y0 = [data.x[0], data.y[0]],
                    params=params_fit, consts=consts_fit, tmax=len(treatment_schedule), dt=1).simulate()
        ax.plot(data.time, sol[:, 0], color='#5E82B8', lw=2, ls="-.", markersize=12, label="X (Median)")
        ax.plot(data.time, sol[:, 1], color='#EBAA42', lw=2, ls="-.", markersize=14, label="Y (Median)")
        ax.set_xlim([0, 40])
        # ax.plot(data.time, sol[:, 0] + sol[:, 1], color="k", lw=2, ls="-.", markersize=14, label="Total (Median)")
        # ax.legend()
        fig.set_tight_layout(True)
        fig.savefig('./plots/mtd_poster.pdf', transparent=True)


if __name__ == '__main__':

    os.chdir('/home/saif/Projects/PhysiLearning')

    ############################# MTD data #############################
    # Plate3 D2 MTd
    df_mtd = pd.read_hdf(
        './data/29112024_2d_manuals/mtd_demo/Evaluations/PcEnvEval_job_1388539920241129_2D_manuals_mtd_demo.h5',
        key='run_0')  # df_mtd2 = pd.read_hdf('./Evaluations/3d_mtd.h5')

    data_list = [df_mtd]

    iteration = 1
    accuracy = 0.0
    tune_draws = 1000
    final_draws = 1000
    consts_fit = {'r_s': 0.045, 'delta_s': 0.01, 'delta_r': 0.01, 'c_s': 4.82, 'c_r': 1.0, 'K': 18000, 'Delta_r': 0.0,
                  'Delta_s': 367.0}
    params_fit = {'r_r': 0.134}
    sigmas = [10, 0.05, 0.01]
    sim_end = df_mtd.index[np.where(~df_mtd.any(axis=1))[0][0]]
    treatment_schedule = np.array([np.int32(i) for i in np.array(df_mtd['Treatment'].values[0:sim_end:1])])
    # shfit by 1 to the right
    # treatment_schedule = np.roll(treatment_schedule, -1)

    trace = az.from_json('./data/SI_data/2D_29112024_mtd_LV.json')
    sampler = "DEMetropolis"
    plot_finals()
    plt.show()

    #res = run_model(theta=[0.1, 0.1, 0.1], y0=[data.x[0], data.y[0]], treatment=df['Treatment'].values)