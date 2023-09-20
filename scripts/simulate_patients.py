# imports
from physilearning.envs import PcEnv
from physilearning.evaluate import Evaluation
import click
import os
import yaml


@click.command()
@click.option('--jobid', default=0, help='ID of the job')
@click.option('--port', default=0, help='ID of the task')
def main(jobid, port):

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_type = config['eval']['evaluate_on']
    save_name = config['eval']['save_name']
    fixed = config['eval']['fixed_AT_protocol']
    at_type = config['eval']['at_type']
    threshold = config['eval']['threshold']

    env = PcEnv.from_yaml('config.yaml', port=str(port), job_name=str(jobid))
    evaluation = Evaluation(env)
    evaluation.run_environment(model_name='None', num_episodes=11,
                               save_path=os.path.join('.', 'Evaluations'),
                               save_name=env_type + 'Eval' + save_name, fixed_therapy=fixed,
                               fixed_therapy_kwargs={'at_type': at_type, 'threshold': threshold})


if __name__ == '__main__':
    main()
