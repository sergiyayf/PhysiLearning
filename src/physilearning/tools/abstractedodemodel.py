import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from physilearning.envs import SLvEnv

class ODEModel:
    """
    ODE model class for the Lottka-Volterra equations or other simulatuion models.

    :param y0: Initial conditions for the state variables. Array-like of length 2.
    :param params: Parameter values for the Lotka-Volterra equations. Array-like of length 4.
    :param tmin: Minimum time value for the simulation. Default is 0.
    :param tmax: Maximum time value for the simulation. Default is 100.
    :param dt: Time step for the ODE solver. Default is 0.1.
    :param treatment_schedule: Time at which to apply the treatment. Default is None.
    :param time: Time points at which to evaluate the solution. Default is None.

    :ivar rhs: The right-hand side of the ODE system.

    """
    def __init__(
            self,
            y0: list or np.ndarray or tuple = (50.45, 1.005),
            params: dict =  {'r_s': 0.03, 'r_r': 0.03, 'delta_s': 0.0003, 'delta_r': 0.0003},
            consts: dict = {'c_s': 1, 'c_r': 1, 'K': 3906, 'Delta_r': 0, 'Delta_s': 5.15},
            tmin: int = 0,
            tmax: int = 100,
            dt: float = 0.1,
            treatment_schedule: list or np.array = None,
            time: list or np.array = None,
            theta: list = [0.03, 0.03, 0.0003, 0.0003],
            const_values = [1, 1, 1, 1, 1.5, 0, 0.15],
    ) -> None:

        self.y0 = y0
        self.params = params
        if time is None:
            self.time = np.arange(tmin, tmax, dt)
        else:
            self.time = time
        self.dt = dt
        if treatment_schedule is None:
            self.treatment_schedule = np.zeros(len(self.time))
        else:
            self.treatment_schedule = self._prep_treatment_schedule(treatment_schedule)
        self.intervals = self.get_treatment_intervals()
        self.treatment = 0
        self.const = consts
        self.theta = theta
        self.env = SLvEnv().from_yaml('/home/saif/Projects/PhysiLearning/config.yaml')

    def update_env_params(self, theta):
        prm = {}
        for key, value in zip(self.params.keys(), theta):
            prm[key] = value
        for key, value in self.const.items():
            prm[key] = value

        self.env.growth_rate = [prm['r_s'], prm['r_r']]
        self.env.death_rate = [prm['delta_s'], prm['delta_r']]
        self.env.death_rate_treat = [prm['Delta_s'], prm['Delta_r']]
        self.env.capacity = prm['K']
        self.env.competition[1] = prm['c_r']
        self.env.competition_exponent = prm['c_s']


    def _prep_treatment_schedule(self, treatment_schedule):
        # check if treatment_schedule items are int32
        if not all(isinstance(item, np.int32) for item in treatment_schedule):
            treatment_schedule = [np.int32(i) for i in treatment_schedule]
        return treatment_schedule

    def get_treatment_intervals(self):
        """Get treatment intervals from treatment schedule

        :return: Array of intervals where treatment is applied
        """
        if self.treatment_schedule is None:
            return None
        else:
            # get indices where treatment changes
            indices = np.where(np.diff(self.treatment_schedule) != 0)[0] + 1
            # add len of treatment_schedule to indices at the end
            indices = np.append(indices, len(self.treatment_schedule))
            indices = np.insert(indices, 0, 0)

            # create intervals of consecutive indices
            intervals = np.stack((indices[:-1], indices[1:]), axis=-1)
            return intervals


    def simulate(self):
        """
        Solve the Lotka-Volterra model for the given time points.

        :return: The solution to the Lotka-Volterra model at the given time points.
        """

        # define solution array to append to
        #solution = np.array([self.y0])
        # update env parameters:
        self.update_env_params(self.theta)
        self.env.reset()
        solution = np.array([self.env.state[0:2]])
        #print(self.env.state)
        for action in self.treatment_schedule[:-1]:
            self.env.step(action)
            #print(self.env.state)
            x, y = self.env.state[0:2]
            solution = np.append(solution, [[x, y]], axis=0)

        return solution


    def plot_model(self, ax=None, solution=None, alpha=0.8, lw=2, title="Lotka-Volterra Model"):
        """
        Plot the Lotka-Volterra model.

        :param ax: Axis on which to plot the model.
        :param solution: Solution to the Lotka-Volterra model.
        :param alpha: Alpha value for the plot.
        :param lw: Line width for the plot.
        :param title: Title for the plot.
        :return: Axis with the plot.
        """

        if solution is None:
            x_y = self.simulate()
        else:
            x_y = solution
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.time, x_y[:, 0], color="b", alpha=alpha, lw=lw, label="x (Model)")
        ax.plot(self.time, x_y[:, 1], color="g", alpha=alpha, lw=lw, label="y (Model)")
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)
        return ax
#
#
# if __name__ == "__main__":
#
#     treatment_schedule = [0, 0, 0, 1, 0, 0, 0, 0, 0]
#     model = ODEModel(treatment_schedule=treatment_schedule, dt=1, tmax=10)
#     sol = model.simulate()
#     fig, ax = plt.subplots()
#     ax = model.plot_model(solution=sol)
#     plt.show()