import multiprocessing as mp
import subprocess
import time
def worker(port):
    command = "srun --exclusive --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3000 run.sh "+port+"&"
    p = subprocess.Popen([command], shell=True)
    return

if __name__ == '__main__':
    num_workers = 2
    for i in range(num_workers):
        worker(str(i))

time.sleep(120)
print('Finished whatever')
