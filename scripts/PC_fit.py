import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.odemodel import ODEModel
from physilearning.tools.bayes import ODEBayesianFitter
import arviz as az

def pcfit():
    plt.rcParams.update({'font.size': 20,
                             'font.weight': 'normal',
                             'font.family': 'sans-serif'})
    mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
    #mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial


    df = pd.read_csv(
        './../Evaluations/older_evals/0_AT_fixedAT_60onPC.csv',
        index_col=[0])

    data = pd.DataFrame(dict(
            time=df.index.values[0:230:1],
            x=df['Type 0'].values[0:230:1],
            y=df['Type 1'].values[0:230:1],))

    treatment_schedule = [np.int32(i) for i in np.array(df['Treatment'].values[0:230:1])]

    print(data)
    print(treatment_schedule)
    y0 = [data.x[0], data.y[0]]
    params = [0.0357, 0.03246, 0.00036, 0.00036]
    model = ODEModel(time=data.time, treatment_schedule=treatment_schedule, dt=1, y0=y0, params=params)
    solution = model.simulate()
    fig, ax = plt.subplots(figsize=(12, 8))
    model.plot_model(ax=ax, solution=solution)
    ax.plot(data.time, data.x, color="b", lw=1, marker="o", markersize=5, label="x data")
    ax.plot(data.time, data.y, color="g", lw=1, marker="+", markersize=5, label="y data")
    ax.legend()

    bayes_fitter = ODEBayesianFitter(model, data)
    # likelihood = bayes_fitter.likelihood(bayes_fitter.set_priors())
    trace = bayes_fitter.sample(draws=50, chains=8)
    print(az.summary(trace))
    fig, ax = plt.subplots(figsize=(12, 8))
    # plot_inference(ax, trace, num_samples=25)
    bayes_fitter.plot_inference_trace(ax=ax, num_samples=25, alpha=0.2)
    ax.plot(data.time, data.x, color="b", lw=2, marker="o", markersize=5, label="x data")
    ax.plot(data.time, data.y, color="g", lw=2, marker="+", markersize=5, label="y data")

    plt.show()


def bayes_main():
    # treatment_schedule = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])

    treatment_schedule = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    model = ODEModel(tmax=22, treatment_schedule=treatment_schedule, dt=1)

    time = model.time
    solution = model.solve(model.time)
    sol2 = model.simulate()

    model.plot_model(solution=solution)
    model.plot_model(solution=sol2)
    fig, ax = plt.subplots()
    model.plot_model(ax=ax, solution=sol2)

    noise = np.random.normal(0, 0.001, size=solution.shape)
    sol2 += noise
    ax.scatter(time, sol2[:, 0], label='x')
    ax.scatter(time, sol2[:, 1], label='y')

    treatment_schedule = [np.int32(i) for i in treatment_schedule]
    ode_model = ODEModel(tmax=22, treatment_schedule=treatment_schedule, dt=1)
    x = sol2[:, 0]
    y = sol2[:, 1]
    data = pd.DataFrame(dict(
        time=time,
        x=x,
        y=y))
    bayes_fitter = ODEBayesianFitter(ode_model, data)
    # likelihood = bayes_fitter.likelihood(bayes_fitter.set_priors())
    trace = bayes_fitter.sample(draws=100, chains=8, cores=16)
    print(az.summary(trace))
    fig, ax = plt.subplots(figsize=(7, 4))
    # plot_inference(ax, trace, num_samples=25)
    bayes_fitter.plot_inference_trace(ax=ax, num_samples=25, alpha=0.2)
    ax.plot(data.time, data.x, color="b", lw=2, marker="o", markersize=12, label="x")
    ax.plot(data.time, data.y, color="g", lw=2, marker="+", markersize=14, label="y")

    plt.show()


def main():
    bayes_main()


if __name__ == '__main__':
    pcfit()