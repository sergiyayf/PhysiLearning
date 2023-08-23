import os.path

import yaml
import subprocess
import re
from physilearning.tools.xml_reader import CfgRead
import click
from typing import Dict


def change_pc_config(pc_conf: Dict, n_envs: int = 1):
    """
    Change the PhysiCell config file with the parameters specified
    under the PC env in the config.yaml file.

    :param pc_conf: dictionary with the parameters to change
    :param n_envs: number of environments to run in parallel
    """
    remove_old_simulation_folders = 'bash ./scripts/cleanup_simulations.sh'
    subprocess.call([remove_old_simulation_folders], shell=True)

    copy_physicell_source = 'bash ./scripts/create_dirs.sh {0}'.format(n_envs - 1)
    subprocess.call([copy_physicell_source], shell=True)
    for i in range(n_envs):
        xml_reader = CfgRead(f'./simulations/PhysiCell_{i}/config/PhysiCell_settings.xml')

        for key in pc_conf:
            print('Changing {0} to {1}'.format(key, pc_conf[key]['value']))
            xml_reader.write_new_param(parent_nodes=pc_conf[key]['parent_nodes'], parameter=key,
                                       value=pc_conf[key]['value'])


@click.group()
def cli():
    pass


@cli.command()
def train():
    """Submit a training job. The job specs are defined in the config.yaml
    under the job section. These define the number of nodes, max time of the job
    etc.

    Job specs:
    ~~~~~~~~~~
    nodes: number of nodes to use
    ntasks: number of tasks to run
    cpus-per-task: number of cpus per task
    mem-per-task: memory per task in MB
    time: wall clock time

    Environment type, their number are defined under the env section.

    This function will call the scripts/job.sh script to submit the SLURM job
    that will execute the physilearning.train.train function.
    """

    # read config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get essential job options
    nodes = config['job']['nodes']
    ntasks = config['job']['ntasks']
    cpus_per_task = config['job']['cpus-per-task']
    mem_per_task = config['job']['mem-per-task']
    mem = mem_per_task * cpus_per_task
    wall_clock_time = config['job']['time']
    n_envs = config['env']['n_envs']
    agent_buffer = config['job']['agent_buffer']
    cpus_per_task += agent_buffer
    if cpus_per_task < agent_buffer + n_envs:
        print('Warning: Too few CPUs allocated for the job')

    # prepare PhysiCell simulations for job submission
    if config['env']['type'] == 'PcEnv':
        pc_conf = config['env']['PC']
        change_pc_config(pc_conf, n_envs)

    # construct a command to run by shell
    if config['global']['machine'] == 'raven':
        job_script = 'raven_job.sh'
    else:
        job_script = 'job.sh'
    command = f'cd ./scripts && sbatch --nodes={nodes} --ntasks={ntasks} --mem={mem}MB --cpus-per-task={cpus_per_task} \
                --time={wall_clock_time} {job_script}'
    p = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
    print(str(out, 'utf-8'))
    # get submitted job ID in case want to directly evaluate the job after job finishes
    # for recurrent jobs, and for communication ports to avoid conflicts with multiple environments
    jobid = re.findall(r'\d+', str(out, 'utf-8'))[0]
    copy_command = 'cp config.yaml config_{0}.yaml'.format(jobid)
    subprocess.call([copy_command], shell=True)

    # cleanup monitor
    monitor_path = os.path.join('Training', 'Logs', 'monitor.csv')
    clean_monitor = f'rm -f {monitor_path}'
    subprocess.call([clean_monitor], shell=True)
    create_monitor = f'cp {os.path.join("Training", "Logs", "empty_monitor.csv")} {monitor_path}'
    subprocess.call([create_monitor], shell=True)

    # run evaluation_job submission script with dependency to run after RL job is finished
    if config['global']['evaluate_after']:
        eval_command = 'cd ./scripts && sbatch --dependency=afterany:{0} evaluation_job.sh'.format(jobid)
        p_eval = subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)
        (out, err) = p_eval.communicate()
        print('Evaluation job: ', str(out, 'utf-8'))

    # run recurrent jobs if specified in config
    if config['job']['recurrent']['enable']:
        for i in range(config['job']['recurrent']['n_jobs']):
            recurrent_command = f'cd ./scripts && sbatch --dependency=afterany:{jobid} --nodes={nodes} \
                                --ntasks={ntasks} \
                                --mem={mem}MB --cpus-per-task={cpus_per_task} \
                                --time={wall_clock_time} {job_script}'
            p_recurrent = subprocess.Popen([recurrent_command], shell=True, stdout=subprocess.PIPE)
            (out, err) = p_recurrent.communicate()
            print('Recurrent job: ', str(out, 'utf-8'))
            jobid = re.findall(r'\d+', str(out, 'utf-8'))[0]
            copy_command = 'cp config.yaml config_{0}.yaml'.format(jobid)
            subprocess.call([copy_command], shell=True)


@cli.command()
def simulate_patients():
    """Submit a job to simulate virtual patients patients
    """
    click.echo('Simulating patients')
    eval_command = 'cd ./scripts && sbatch --nodes=1 --cpus-per-task=10 --ntasks=1 simulate_patients_job.sh'
    subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)


@cli.command()
def evaluate():
    """Submit a job to evaluate the trained model or how fixed AT protocol performs
    """
    # read config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_envs = config['env']['n_envs']
    # prepare PhysiCell simulations for job submission
    if config['eval']['evaluate_on'] == 'PcEnv':
        pc_conf = config['env']['PC']
        change_pc_config(pc_conf, n_envs)
    click.echo('Evaluating')
    eval_command = 'cd ./scripts && sbatch evaluation_job.sh'
    subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)


if __name__ == '__main__':
    cli()
