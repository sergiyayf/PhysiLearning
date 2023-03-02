

class Trainer():
    """Class to train a model.

    """
    def __init__(self,config_file):
        """Initialize the trainer class.

        Args:
        -----
            config_file (str): Path to the config file.

        Parameters:
        -----------
            config (dict): Dictionary containing the configuration parameters.
            model_path (str): Path to the directory where the model will be saved.
            save_freq (int): Frequency at which the model will be saved.
            name_prefix (str): Prefix to the name of the model.
            n_envs (int): Number of environments to be used for training.
            ent_coef (float): Entropy coefficient.
            n_steps (int): Number of steps to be taken in each environment.
            total_timesteps (int): Total number of timesteps to be taken.
            verbose (int): Verbosity level.
            enable_loading (bool): Whether to load a model or not.
            load_from_external_file (bool): Whether to load a model from an external file or not.
            external_file_name (str): Path to the external file from which the model will be loaded.
            env_type (str): Type of environment to be used.
            logname (str): Name of the log file.
            optimization_algorithm (str): Name of the optimization algorithm to be used.
            checkpoint_callback (CheckpointCallback): Callback to save the model.
            copy_config_callback (CustomCallback): Callback to copy the config file to the log directory.
            tensorboard_callback (TensorboardCallback): Callback to save the tensorboard logs.
            env (VecEnv): Vectorized environment.
            model (PPO): Model to be trained.
            log_dir (str): Path to the log directory.
            log_path (str): Path to the log file.
            log_file (file): Log file.
        """
        self.config_file = config_file
        with open(config_file) as f:
            self.config = yaml.load(open(config_file),Loader=yaml.FullLoader)
        self.model_path = os.path.join('Training', 'SavedModels')
        self.save_freq = self.config['learning']['model']['save_freq']
        self.name_prefix = self.config['learning']['model']['model_save_prefix']
        self.n_envs = self.config['learning']['model']['n_envs']
        self.ent_coef = self.config['learning']['model']['ent_coef']
        self.n_steps = self.config['learning']['model']['n_steps']
        self.total_timesteps = self.config['learning']['model']['total_timesteps']
        self.verbose = self.config['learning']['model']['verbose']
        self.enable_loading = self.config['learning']['model']['load']['enable_loading']
        self.load_from_external_file = self.config['learning']['model']['load']['external_file_loading']
        self.external_file_name = self.config['learning']['model']['load']['external_file_name']
        self.env_type = self.config['env']['type']
        self.logname = self.config['learning']['model']['model_save_prefix']
        self.log_path = os.path.join('Training','Logs',self.logname)
        self.model = None
        self.env = None

    def create_callbacks(self):
        """Create callbacks for training.
        """
        self.checkpoint_callback = CheckpointCallback(save_freq=self.save_freq,save_path = self.model_path, name_prefix = self.name_prefix)
        self.copy_config_callback = CustomCallback(self.config_file,self.logname)
        #self.tensorboard_callback = TensorboardCallback()
        self.callback_list = [self.checkpoint_callback,self.copy_config_callback]
        return self.callback_list

    def create_env(self):
        """Create the environment.
        """
        num_cpu = self.n_envs
        if self.n_envs == 1:
            print('Training agent on one environment')
            if self.env_type == 'PhysiCell':
                from physilearning.PC_environment import PC_env
                self.env = PC_env.from_yaml(self.config_file,port='0',job_name=sys.argv[1])
            elif self.env_type == 'LV':
                from physilearning.ODE_environments import LV_env
                self.env = LV_env.from_yaml(self.config_file,port='0',job_name=sys.argv[1])
            elif self.env_type == 'jonaLVenv':
                from physilearning.jonaLVenv import jonaLVenv
                self.env = jonaLVenv()
            else:
                raise ValueError('Environment type not recognized')

