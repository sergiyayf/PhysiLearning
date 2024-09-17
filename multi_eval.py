import os
import sys
import yaml
from physilearning.train import Trainer
from physilearning.evaluate import Evaluation

config_file = 'config.yaml'

for i in range(1, 21):

    with open(config_file, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
        # define paths and load others from config
    print('Parsing config file {0}'.format(config_file))

    if general_config['eval']['from_file']:
        # configure environment and model to load
        model_training_path = general_config['eval']['path']
        model_prefix = f'20240910_tendayaverage_multi_{i}'
        model_config_file = os.path.join(model_training_path, 'Training', 'Configs', model_prefix + '.yaml')

        env_type = general_config['eval']['evaluate_on']
        save_name = general_config['eval']['save_name']
        with open(model_config_file, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        if env_type == 'same':
            env_type = model_config['env']['type']
            train = Trainer(model_config_file)
        else:
            train = Trainer(config_file)
        train.env_type = env_type
        train.setup_env()
        evaluation = Evaluation(train.env)

        model_name = os.path.join(model_training_path, 'Training', 'SavedModels',
                                  model_prefix + general_config['eval']['step_to_load'])
        fixed = general_config['eval']['fixed_AT_protocol']
        at_type = general_config['eval']['at_type']
        print(model_name)
        evaluation.run_environment(model_name, num_episodes=general_config['eval']['num_episodes'],
                                   save_path=os.path.join(model_training_path, 'Evaluations'),
                                   save_name=env_type + 'Eval_' + save_name + model_prefix, fixed_therapy=fixed,
                                   fixed_therapy_kwargs={'at_type': at_type})

    else:
        env_type = general_config['eval']['evaluate_on']
        save_name = general_config['eval']['save_name']
        train = Trainer(config_file)
        train.env_type = env_type
        train.setup_env()
        evaluation = Evaluation(train.env)

        fixed = general_config['eval']['fixed_AT_protocol']
        at_type = general_config['eval']['at_type']
        threshold = general_config['eval']['threshold']
        if env_type == 'PcEnv':
            job_name = 'job_ ' +str(sys.argv[1])
        else:
            job_name = '_'
        evaluation.run_environment(model_name='None', num_episodes=general_config['eval']['num_episodes'],
                                   save_path=os.path.join('.', 'Evaluations'),
                                   save_name=env_type +'Eval_ ' +job_name +save_name, fixed_therapy=fixed,
                                   fixed_therapy_kwargs={'at_type': at_type, 'threshold': threshold})