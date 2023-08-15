from physilearning.train import Trainer
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
import optuna

class HyperparameterTuner:
    """
    HyperparameterTuner is a class that tunes the hyperparameters of an agent
    """
    def __init__(
        self,
        config_file: str = 'config.yaml',
        n_startup_trials: int = 0,
        n_evaluations: int = 100,

    ):
        self.env = None
        self.seed = None
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.config_file = config_file

    def _learn(self):
        """
        _setup_env sets up the environment for the agent to interact with
        """
        trainer = Trainer(config_file=self.config_file)
        trainer.setup_env()
        trainer.learn()


    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner


    def _sample_hyperparams(self, trial: optuna.Trial):
        """
        _sample_hyperparams samples the hyperparameters of the agent
        """
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.01, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])

