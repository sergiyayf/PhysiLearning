# imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

class ODEModel:
    """
    ODE model class for the Lotka-Volterra equations or other simulatuion models.

    Attributes
    ----------
    rhs : function
        The right-hand side of the ODE system.
    y0 : list or array-like
        The initial conditions for the state variables.
    theta : list or array-like
        The parameter values for the Lotka-Volterra equations.
    time : array-like
        The time points at which to evaluate the solution.
    treatment_time : float
        The time at which to apply the treatment.
    dt : float
        The time step for the ODE solver.
    """
    def __init__(self, y0=[0.045, 0.005], params=[0.0357, 0.0325, 0.0003, 0.0003], tmin=0, tmax=100, dt=0.1, treatment_schedule=None, time=None):
        """Initialize the ODEModel class.

        Parameters
        ----------

        y0 : list or array-like
            The initial conditions for the state variables.
        theta : list or array-like
            The parameter values for the Lotka-Volterra equations.
        time : array-like
            The time points at which to evaluate the solution.
        """
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
            self.treatment_schedule = treatment_schedule
        self.intervals = self.get_treatment_intervals()
        self.treatment = 0
        self.const = {'c_x': 1, 'c_y': 1, 'K': 1.5, 'Delta_y': 0, 'Delta_x': 0.15}

    def __call__(self, *args):
        return self.rhs(self.y0, self.time, self.params, *args)

    def get_treatment_intervals(self):
        """Get treatment intervals from treatment schedule

        Returns
        -------
        intervals : array-like
            The intervals of the treatment schedule.
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
    def _LV(self, t, X, theta):
        """
        Define the Lotka-Volterra equations.

        Parameters
        ----------
        t : float
            The current time.
        X : list or array-like
            The current values of the state variables.
        theta : list or array-like
            The parameter values for the Lotka-Volterra equations.

        Returns
        -------
        list
            The derivatives of the state variables at the current time.
        """
        x, y = X
        alpha, beta, gamma, delta = theta
        # equations
        dx_dt = alpha * x - beta * x * y
        dy_dt = -gamma * y + delta * x * y
        return [dx_dt, dy_dt]

    def LV(self, t, X, theta):
        """
        Define the Lotka-Volterra equations.

        Parameters
        ----------
        t : float or array-like
            The current time.
        X : list or array-like
            The current values of the state variables.
        theta : list or array-like
            The parameter values for the Lotka-Volterra equations.

        Returns
        -------
        list
            The derivatives of the state variables at the current time.
        """
        # unpack parameters
        # resistant and susceptible populations
        x, y = X
        # growth and death rates
        r_x, r_y, delta_x, delta_y = theta
        # competition parameters
        c_x = self.const['c_x']
        c_y = self.const['c_y']
        # carrying capacity
        K = self.const['K']
        # treatment death rates
        Delta_y = self.const['Delta_y']
        Delta_x = self.const['Delta_x']

        # equations
        dx_dt = x*(r_x*(1-(x+y*c_y)/K)-delta_x-Delta_x*self.treatment)
        dy_dt = y*(r_y*(1-(y+x*c_x)/K)-delta_y-Delta_y*self.treatment)

        return [dx_dt, dy_dt]

    def simulate(self):
        """
        Solve the Lotka-Volterra model for the given time points.

        Returns
        -------
        array-like
            The solution to the Lotka-Volterra model at the given time points.
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
                solution = np.append(solution,sol[1:],axis=0)
            elif treat == 0:
                self.treatment = 0
                sol = self.solve(time)
                solution = np.append(solution,sol[1:],axis=0)
            self.y0 = solution[-1]

        return solution

    def solve(self, time):
        """
        Solve the Lotka-Volterra model for the given time points.

        Parameters
        ----------
        time : array-like
            The time points at which to evaluate the solution.

        Returns
        -------
        array-like
            The solution to the Lotka-Volterra model at the given time points.
        """

        return solve_ivp(fun=self.rhs,
            y0=self.y0,
            t_span=[time[0], time[-1]],
            args=(self.params,),
            dense_output=True,
        ).sol(time).T

    def plot_model(self, ax=None, solution=None, alpha=0.8, lw=2, title="Lotka-Volterra Model"):
        """
        Plot the Lotka-Volterra model.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            The axis object to use for the plot. If not provided, a new axis object is created.
        time : array-like, optional
            The time points at which to evaluate the solution.
        alpha : float, optional
            The alpha value for the plot lines.
        lw : float, optional
            The line width for the plot lines.
        title : str, optional
            The title of the plot.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            The axis object used for the plot.
        """
        if solution is None:
            x_y = self.solve(self.time)
        else:
            x_y = solution
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.time, x_y[:, 0], color="b", alpha=alpha, lw=lw, label="x (Model)")
        ax.plot(self.time, x_y[:, 1], color="g", alpha=alpha, lw=lw, label="y (Model)")
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(title, fontsize=16)
        return ax

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