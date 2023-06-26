# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Plotter:
    def __init__(self, env):
        self.env = env

    def plot_trajectory(self, trajectory_file, save_name = None):
        """
        Method to plot the trajectory of the tumor cells
        Parameters
        ----------
        trajectory_file: str, path to the trajectory file
        save_name: str, path to save the plot

        Returns
        -------

        """
        fig, ax = plt.subplots(figsize=(8, 4))
        df = pd.read_csv(trajectory_file, index_col=[0])
        x = np.arange(0, len(df))
        ax.fill_between(x, 0, df['Treatment'], color='orange', label='drug')
        ax.plot(x, (df['Type 0'] + df['Type 1']), 'k', label='total', linewidth=2)
        ax.plot(x, df['Type 0'], 'b', label='wt', linewidth=2)
        ax.plot(x, df['Type 1'], 'g', label='mut', linewidth=2)
        ax.set_xlabel('time')
        ax.set_ylabel('# Cells')
        ax.legend()
        # make constrained layout
        fig.tight_layout()
        if save_name is not None:
            fig.savefig(save_name, transparent=True)
        plt.show()

