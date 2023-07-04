import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data(ax, lw=2, title="Initial data"):
    ax.plot(data.time, data.x, color="b", lw=lw, marker="o", markersize=12, label="X (Data)")
    ax.plot(data.time, data.y, color="g", lw=lw, marker="+", markersize=14, label="Y (Data)")
    # fill between when treatment is on
    ax.fill_between(data.time, 0, max(data.x), where=treatment_schedule==1, facecolor='orange', alpha=0.5, label="Treatment")

    ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title, fontsize=16)
    return ax

if __name__ == '__main__':

    # Get data
    df = pd.read_csv(
        './Evaluations/0_PcEnvEvalpatient_18_fixed_06.csv',
        index_col=[0])

    data = pd.DataFrame(dict(
            time=df.index.values[0:56:1],
            x=df['Type 0'].values[0:56:1],
            y=df['Type 1'].values[0:56:1],))


    treatment_schedule = np.array(df['Treatment'].values[0:56:1])
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="Original treatment")
    # append treatmentd schedule with two zeros from the beginning, delete last 2
    treatment_schedule = [np.int32(i) for i in np.array(df['Treatment'].values[0:56:1])]
    treatment_schedule = np.array([0] + treatment_schedule[:-1])
    # find ends of treatment
    treatment_ends = np.where(np.diff(treatment_schedule) == -1)[0]
    # replace ends of treatment with 1
    treatment_schedule[treatment_ends+1] = 1


    # Plot data
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_data(ax, title="Shifted treatment")

    plt.show()