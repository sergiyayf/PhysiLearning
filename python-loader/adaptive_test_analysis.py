from load.matTohdf import matTohdf5
import time
from multiprocessing import Pool
import warnings
import sys
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
start_time = time.time()

dir_path = sys.argv[1]
print('dir_path this ',dir_path)

def process_data(filename):
    matTohdf5(filename);
    return 0;


def get_directories_list():
    dir_list = [];
    for i in range(1, 2):
        directory_path = dir_path
        dir_list.append(directory_path);

    return dir_list;

def analysis_stuff():
    filename = dir_path + '/all_data.h5';
    f = h5py.File(filename, 'r')

    dat = f['data/cells'];
    cells_look_info = pd.read_hdf(filename, 'data/cells/cell1');
    print(cells_look_info)

    read_radii = pd.read_hdf(filename, 'data/radius')
    # print(read_radii)
    fronts = pd.read_hdf(filename, 'data/front_cell_number')
    # print(fronts)
    read_growth_layer = pd.read_hdf(filename, 'data/growth_layer');
    # print(read_growth_layer);
    Rad = np.array(read_radii)
    Growth = np.array(read_growth_layer);
    num_cells = np.array(fronts);

    frequency = num_cells / np.sum(num_cells, 1)[:, None]
    print('frequency = ', frequency)
    types = ['wt', 'rd', 'rn']
    color_order = 'yrc'
    plt.figure()
    plt.plot(Rad,num_cells)
    plt.legend(types)
    plt.figure()
    plt.plot(Rad,frequency)
    plt.legend(types)

    return frequency[-1]
def write_output(input, output):
    with open('output.csv', 'a') as f:
        f.write(str(input) + ',' + str(output) + '\n')
    return
def save_frequency(input,output):
    write_output(input,output)
    return
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    list_of_directories = get_directories_list();
    with Pool(4) as p:
        p.map(process_data, list_of_directories);
        print('anything')

    print('Done all!')
    freq = analysis_stuff()
    print('Analysis stuff done')
    save_frequency(1,freq[0])
    print('frequency_saved')


    print(time.time() - start_time)
    plt.show()
