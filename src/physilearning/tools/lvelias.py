import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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
            y0: list or np.ndarray or tuple = (0.045, 0.005),
            params: dict =  {'r_s': 0.07, 'r_r': 0.07, 'Delta': 5.0},
            consts: dict = {'K_s': 120000, 'K_r': 120000},
            tmin: int = 0,
            tmax: int = 100,
            dt: float = 0.1,
            treatment_schedule: list or np.array = None,
            time: list or np.array = None,
            theta: list = [0.07, 0.07, 5.0],
            const_values = [120000],
    ) -> None:
        self.rhs = self.LV
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

    def __call__(self, *args):
        return self.rhs(self.y0, self.time, self.params)


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

    def LV(self, t, X, theta):
        """
        Define the Lotka-Volterra equations.

        :param t: Time points at which to evaluate the solution.
        :param X: State variables.
        :param theta: Parameters for the Lotka-Volterra equations.
        :return: The right-hand side of the Lotka-Volterra equations.

        """
        # unpack resistant and susceptible populations
        s, r = X
        prm = {}
        for key, value in zip(self.params.keys(), theta):
            prm[key] = value
        for key, value in self.const.items():
            prm[key] = value

        # check if all parameters are specified in either theta or self.const
        list_of_all_parameters = ['r_s', 'r_r', 'K', 'Delta_s', 'K']
        for parameter in list_of_all_parameters:
            if parameter not in prm:
                raise ValueError('Parameter {} not found in parameters dictionary'.format(parameter))

        # equations
        if prm['r_r'] == 1.25:
            prm['r_r'] = 1.25*prm['r_s']
        dx_dt = s*(prm['r_s']*(1-(s+r)/prm['K'])*(1-self.treatment*prm['Delta_s']) - 0.1*prm['r_s'])
        dy_dt = r*(prm['r_r']*(1-(r+s)/prm['K'])-0.1*prm['r_r'])

        return [dx_dt, dy_dt]

    def simulate(self):
        """
        Solve the Lotka-Volterra model for the given time points.

        :return: The solution to the Lotka-Volterra model at the given time points.
        """

        # define solution array to append to
        solution = np.array([self.y0])
        # get treatment intervals to split the simulation into treatment and no treatment regions
        intervals = self.get_treatment_intervals()
        # loop over intervals
        for interval in intervals:
            # get treatment value for interval (on or off)
            treat = self.treatment_schedule[interval[0]]
            # define time points for interval, if last interval, don't include last time point
            if interval[1] == intervals[-1][1]:
                time = np.arange(interval[0], interval[1], self.dt)
            else:
                time = np.arange(interval[0], interval[1]+self.dt*0.001, self.dt)

            # solve for interval depending on treatment value, changing attribute self.treatment
            if treat == 1:
                self.treatment = 1
                sol = self.solve(time)
                solution = np.append(solution, sol[1:], axis=0)
            elif treat == 0:
                self.treatment = 0
                sol = self.solve(time)
                solution = np.append(solution, sol[1:], axis=0)
            self.y0 = solution[-1]
        # replace all negative values with 0
        solution[solution < 0] = 0

        return solution[0::2]

    def solve(self, time):
        """
        Solve the Lotka-Volterra model for the given time points.

        :param time: Time points at which to evaluate the solution.
        :return: The solution to the Lotka-Volterra model at the given time points.

        """

        return solve_ivp(fun=self.rhs,
                         y0=self.y0,
                         t_span=[time[0], time[-1]],
                         args=(self.theta,),
                         dense_output=True,
                         ).sol(time).T

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
            x_y = self.solve(self.time)
        else:
            x_y = solution
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.time[::2] / 2, x_y[:, 0], color="b", alpha=alpha, lw=lw, label="x (Model)")
        ax.plot(self.time[::2] / 2, x_y[:, 1], color="g", alpha=alpha, lw=lw, label="y (Model)")
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)
        return ax
