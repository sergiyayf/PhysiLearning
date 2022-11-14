import yaml 
import subprocess
import time 
import re

if __name__=='__main__':
    """
        Code to submit a reinforcement learning PhysiCell job configured by the config.yaml file
    """
    
    # read config
    with open('config.yaml','r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Get essential job options
    nodes = config['job']['nodes']
    ntasks = config['job']['ntasks']
    cpus_per_task = config['job']['cpus-per-task']
    mem_per_task = config['job']['mem-per-task']
    mem = mem_per_task*cpus_per_task
    wall_clock_time = config['job']['time']
    n_envs = config['learning']['model']['n_envs']
    agent_buffer = config['job']['agent_buffer']

    if cpus_per_task < agent_buffer + n_envs : 
        print('Warning: Too few CPUs allocated for the job')

    # prepare PhysiCell simulations for job submission
    copy_PhysiCell = 'bash ./bin/create_dirs.sh {0}'.format(n_envs-1)
    subprocess.call([copy_PhysiCell],shell=True)

    # construct a command to run by shell
    command = 'cd ./bin && sbatch --nodes={0} --ntasks={1} --mem={2}MB --cpus-per-task={3} \
            --time={4} job.sh'.format(nodes,ntasks,mem,cpus_per_task,wall_clock_time)
    p = subprocess.Popen([command], shell = True, stdout=subprocess.PIPE)   
    (out,err) = p.communicate()
    print(str(out,'utf-8'))
    # get submitted job ID in case want to directly evaluate the job after job finishes 
    jobid = re.findall(r'%s(\d+)' % "job", str(out,'utf-8'))
    
    # run evaluation_job submission script with dependency to run after RL job is finished 
    if config['global']['evaluate_after']:
        eval_command = 'cd ./bin && sbatch --dependency=afterany:{0} evaluation_job.sh'.format(jobid)
        p_eval.subprocess.Popen([eval_command],shell=True, stdout=subprocess.PIPE)
        (out,err) = p_eval.communicate()
        print('Evaluation job: ', str(out,'utf-8'))
        
    
