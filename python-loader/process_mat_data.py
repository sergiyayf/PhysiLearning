from load.matTohdf import matTohdf5
import time 
from multiprocessing import Pool
import warnings

start_time = time.time()

#for i in range(1,11):
    #directory_path = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210524_cobra\\19_05_cobra_switch_at_5_run_'+str(i)+'\output'
    #matTohdf5(directory_path)

    #print('Done!')

def process_data(filename): 
    matTohdf5(filename);
    return 0; 

def get_directories_list(): 
    dir_list = [];
    for i in range(1,11):
        directory_path = 'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\\20210524_cobra\\19_05_cobra_switch_at_20_run_'+str(i)+'\output'
        dir_list.append(directory_path); 
    
    return dir_list;

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    list_of_directories = get_directories_list();
    with Pool(4) as p:
        p.map(process_data, list_of_directories);
        print('anything')
        

print('Done all!')
print(time.time()-start_time)
