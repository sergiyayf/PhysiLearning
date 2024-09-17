from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import os


class CopyConfigCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, config_file: str = 'config.yaml', logname: str = 'config', verbose: bool = 0):
        super(CopyConfigCallback, self).__init__(verbose)
        self.logname = logname
        self.config_file = config_file

    def _on_training_start(self) -> bool:
        """
        This method is called before the first rollout starts.
        """
        # copy config.yaml to the training folder using os
        os.system(r'echo "copying config.yaml to Training/Configs/"')
        command = f'cp {self.config_file} ./Training/Configs/{self.logname}.yaml'
        os.system(command)
        #command = f'rm {self.config_file}'
        #os.system(command)

        return True

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str,
                 save_dir: str, save_name: str, verbose: int = 1, average_steps: int = 10):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(save_dir)
        self.save_name = save_name
        self.average_steps = average_steps
        try:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                if len(y) < 5:
                    self.best_mean_reward = np.mean(y[-self.average_steps:]) / 2
                else:
                    self.best_mean_reward = np.mean(y[-self.average_steps:])
            else:
                self.best_mean_reward = -np.inf

        except FileNotFoundError:
            self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 20 episodes
                if len(y) < 5:
                    mean_reward = np.mean(y[-self.average_steps:]) / 2
                else:
                    mean_reward = np.mean(y[-self.average_steps:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} "
                          f"- Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward >= self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                        self.model.save(os.path.join(self.save_path, f'{self.save_name}_best_reward'))

        return True
