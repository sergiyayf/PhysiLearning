import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from physilearning.tools.odemodel import ODEModel
from physilearning.tools.bayes import ODEBayesianFitter

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
#likelihood = bayes_fitter.likelihood(bayes_fitter.set_priors())
# trace = bayes_fitter.sample(draws=5000, chains=8)
# print(az.summary(trace))
# fig, ax = plt.subplots(figsize=(12, 8))
# # plot_inference(ax, trace, num_samples=25)
# bayes_fitter.plot_inference_trace(ax=ax, num_samples=25, alpha=0.2)
# ax.plot(data.time, data.x, color="b", lw=2, marker="o", markersize=5, label="x data")
# ax.plot(data.time, data.y, color="g", lw=2, marker="+", markersize=5, label="y data")

plt.show()
