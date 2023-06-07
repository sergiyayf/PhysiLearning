import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from physilearning.tools.odemodel import ODEModel

if __name__ == "__main__":
    # create an instance of the ODEModel class
    # solve the model

    treatment_schedule = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    model = ODEModel(tmax=22, treatment_schedule=treatment_schedule,dt=1)

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
    plt.show()