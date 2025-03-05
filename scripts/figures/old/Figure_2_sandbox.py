import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def get_ttp(df):
    initial_size = df['Type 0'][0] + df['Type 1'][0]
    nz = df[((df['Type 1']) / initial_size >= 1.0)]
    tot_idx = df[((df['Type 0'] + df['Type 1']) / initial_size >= 2.0)].index
    if len(nz) > 0:
        # append index when type 0 + type 1 is larger than 1.5
        return nz.index[0] / 2
    elif len(tot_idx) > 0:
        return tot_idx[0] / 2
    else:
        return len(df) / 2

def get_average_cell_number(df, type='Total'):
    if type == 'Total':
        tot = df['Type 0'] + df['Type 1']
    else:
        tot = df[type]
    tot = tot[tot > 0]
    average_total = np.mean(tot[:])
    return average_total

def get_lowest_sensitive(df):
    sensitive_cells = df['Type 0']
    #remove zeros if there
    sensitive_cells = sensitive_cells[sensitive_cells > 0]
    lowest_sensitive = np.min(sensitive_cells)
    return lowest_sensitive

def get_largest_sensitive(df):
    sensitive_cells = df['Type 0']
    largest_sensitive = np.max(sensitive_cells)
    return largest_sensitive

def get_lowest_maximum(df):
    y = np.array(df['Type 0'])
    maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    maxima_indices = np.where(maxima)[0] + 1  # Adjust index due to slicing
    m = y[maxima_indices]
    return np.min(m)

def get_index_of_lowest_sensitive(df):
    sensitive_cells = df['Type 0']
    #remove zeros if there
    sensitive_cells = sensitive_cells[sensitive_cells > 0]
    lowest_sensitive = np.min(sensitive_cells)
    index = np.where(df['Type 0'] == lowest_sensitive)
    return index

def average_low_before_progression(df):
    y = np.array(df['Type 0'])
    minima = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    minima_indices = np.where(minima)[0] + 1  # Adjust index due to slicing
    m = y[minima_indices]
    # average of last 5 minima
    if len(m) > 10:
        return np.mean(m[-10:])
    else:
        return np.mean(m)

def average_before_progression(df, type='Type 0'):
    if type == 'Total':
        tot = df['Type 0'] + df['Type 1']
    else:
        tot = df[type]
    tot = tot[tot > 0]
    average_total = np.mean(tot[-30:-10])
    return average_total


def analyze_data(list_of_files):
    #list_of_files=[f'./Evaluations/physicell_0901_tain_6_data/run_{i}/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for i in range(1,11)]
    #list_of_files = [f'./Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5' for i in range(1,11)]
    file_idx = 0
    data_dict = {}
    averaged_data = {}
    std_data = {}
    for file in list_of_files:
        ttps = []
        avgs = []
        for run in range(50):
            df = pd.read_hdf(file, key=f'run_{run}')
            ttp = get_ttp(df)
            avg = get_average_cell_number(df)
            low = get_lowest_sensitive(df)
            high = get_largest_sensitive(df)
            low_max = get_lowest_maximum(df)
            index_low = get_index_of_lowest_sensitive(df)
            avg_low = average_low_before_progression(df)
            before_progression = average_before_progression(df)
            #print(f'Run {run} TTP: {ttp}, AVG: {avg}, LOW: {low}, HIGH: {high}, LOW_MAX: {low_max}')
            data_dict[f'{file_idx}_{run}'] = {'TTP': ttp, 'AVG': avg, 'LOW': low, 'HIGH': high,
                                              'LOW_MAX': low_max, 'INDEX_LOW': index_low,
                                              'AVG_LOW': avg_low, 'BEFORE': before_progression}
            ttps.append(ttp)
            avgs.append(avg)
        averaged_data[file_idx] = {'TTP': np.mean(ttps), 'AVG': np.mean(avgs)}
        std_data[file_idx] = {'TTP': np.std(ttps), 'AVG': np.std(avgs)}
        file_idx += 1
    return data_dict, averaged_data, std_data



if __name__ == '__main__':
    os.chdir('/')
    # list_of_files_pc = [
    #     f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5'
    #     for i in range(1, 11)]
    # data_dict_pc, average_data_pc, std_data_pc = analyze_data(list_of_files_pc)
    # list_of_files_lv = [f'./Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
    #                  for i in range(1, 11)]
    # data_dict_lv, average_data_lv, std_data_lv = analyze_data(list_of_files_lv)

    list_of_files_pc = [
        f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5'
        #f'./Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
        for i in range(1, 11)]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for run in range(0,10):
        df = pd.read_hdf(list_of_files_pc[run], key=f'run_{7}')
        total_cell_number = df['Type 0'] + df['Type 1']
        total_cell_number = total_cell_number[total_cell_number > 0]


        # for each time point plot the sum until this point
        sm = np.cumsum(total_cell_number)
        length = 50
        sm = np.convolve(sm, np.ones(length), 'valid')/length
        # subtract x
        sm = sm - np.arange(len(sm))
        ax.plot(sm, label=f'Run {run}')

        # plot slope of sm
        slope = np.diff(sm)
        ax2.plot(slope, label=f'Slope {run}')

    plt.show()

