import yaml 
import subprocess
import re
from physilearning.tools.xml_reader import CfgRead
import click

def change_PC_config(PC_conf = None, n_envs = 1):
    """Change the PhysiCell settings file to run multiple simulations in parallel
    for now only works for 1 environment
    """
    clean_sims = 'bash ./scripts/cleanup_simulations.sh'
    subprocess.call([clean_sims], shell=True)

    copy_PhysiCell = 'bash ./scripts/create_dirs.sh {0}'.format(n_envs - 1)
    subprocess.call([copy_PhysiCell], shell=True)
    xml_reader = CfgRead('./simulations/PhysiCell_V_1.10.4_0/config/PhysiCell_settings.xml')

    for key in PC_conf:
        print('Changing {0} to {1}'.format(key, PC_conf[key]['value']))
        xml_reader.write_new_param(parent_nodes=PC_conf[key]['parent_nodes'], parameter=key,
                                   value=PC_conf[key]['value'])

    # xml_reader.write_new_param(parent_nodes=['save', 'full_data'], parameter="enable", value='true')

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

    if cpus_per_task < agent_buffer + n_envs:
        print('Warning: Too few CPUs allocated for the job')

    # prepare PhysiCell simulations for job submission
    if config['env']['type'] == 'PcEnv':
        PC_conf = config['env']['PC']
        change_PC_config(PC_conf, n_envs)


    # construct a command to run by shell
    command = 'cd ./scripts && sbatch --nodes={0} --ntasks={1} --mem={2}MB --cpus-per-task={3} \
                --time={4} job.sh'.format(nodes, ntasks, mem, cpus_per_task, wall_clock_time)

    p = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE)
    (out, err) = p.communicate()
    print(str(out, 'utf-8'))
    # get submitted job ID in case want to directly evaluate the job after job finishes
    # jobid = re.findall(r'%s(\d+)' % "job", str(out,'utf-8'))
    # find a number after "Submitted batch job" in the output
    jobid = re.findall(r'\d+', str(out, 'utf-8'))[0]
    # copy config to file config_jobid.yaml
    copy_command = 'cp config.yaml config_{0}.yaml'.format(jobid)
    subprocess.call([copy_command], shell=True)

    # run evaluation_job submission script with dependency to run after RL job is finished
    if config['global']['evaluate_after']:
        eval_command = 'cd ./scripts && sbatch --dependency=afterany:{0} evaluation_job.sh'.format(jobid)
        p_eval = subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)
        (out, err) = p_eval.communicate()
        print('Evaluation job: ', str(out, 'utf-8'))

@cli.command()
def simulate_patients():
    """Submit a job to simulate virtual patients patients
    """
    click.echo('Simulating patients')
    eval_command = 'cd ./scripts && sbatch simulate_patients_job.sh'
    p_eval = subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)

@cli.command()
def evaluate():
    """Submit a job to evaluate the trained model or how fixed AT protocol performs
    """
    # read config
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_envs = config['env']['n_envs']
    # prepare PhysiCell simulations for job submission
    if config['eval']['evaluate_on'] == 'PhysiCell':
        PC_conf = config['env']['PC']
        change_PC_config(PC_conf, n_envs)
    click.echo('Evaluating')
    eval_command = 'cd ./scripts && sbatch evaluation_job.sh'
    p_eval = subprocess.Popen([eval_command], shell=True, stdout=subprocess.PIPE)

if __name__=='__main__':
    cli()

    
