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
    list_of_files_lv = [
        f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5'
        for i in range(1, 11)]
    #data_dict_pc, average_data_pc, std_data_pc = analyze_data(list_of_files_pc)
    list_of_files_lv_ = [f'./Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
                     for i in range(1, 11)]
    data_dict_lv, average_data_lv, std_data_lv = analyze_data(list_of_files_lv)
    # scatter avg vs ttp, low vs ttp, high vs tpp and low_max vs ttp
    fig, ax = plt.subplots(figsize = (10,10))

    for key in average_data_lv.keys():
        # scatter plot with error bars (average_data_lv[key]['AVG'], average_data_lv[key]['TTP'], label=f'LV {key+1}')
        ax.errorbar(average_data_lv[key]['AVG'], average_data_lv[key]['TTP']/2,
                       xerr=std_data_lv[key]['AVG'], yerr=std_data_lv[key]['TTP']/2, fmt='o', color = 'k')
    #ax2.scatter([average_data[key]['AVG'] for key in average_data.keys()], [average_data[key]['TTP'] for key in average_data.keys()])

    #ax.set_title('LV-LV')
    #ax.set_xlim(0.95, 1.05)
    #ax.set_ylim(0, 160)
    #ax.set_ylabel('Time to progression')

    # for key in average_data_lv.keys():
    #     ax[1].errorbar(average_data_lv[key]['AVG'], average_data_pc[key]['TTP'],
    #                  xerr=std_data_lv[key]['AVG'], yerr=std_data_pc[key]['TTP'], fmt='o')
    # # ax2.scatter([average_data[key]['AVG'] for key in average_data.keys()], [average_data[key]['TTP'] for key in average_data.keys()])
    #
    # ax[1].set_title('LV-PC')
    # ax[1].set_xlim(0.95, 1.05)
    # ax[1].set_ylim(0, 320)
    #
    # for key in average_data_lv.keys():
    #     ax[2].errorbar(average_data_pc[key]['AVG'], average_data_pc[key]['TTP'],
    #                  xerr=std_data_pc[key]['AVG'], yerr=std_data_pc[key]['TTP'], fmt='o', label=f'Agent {key+1}')
    # # ax2.scatter([average_data[key]['AVG'] for key in average_data.keys()], [average_data[key]['TTP'] for key in average_data.keys()])
    #
    # ax[2].set_title('PC-PC')
    # ax[2].set_xlim(0.95, 1.05)
    # ax[2].set_ylim(0, 320)
    #
    # # set one x label for all 3 axes
    # fig.text(0.5, 0.04, 'Average cell number', ha='center', va='center')
    # #fig.suptitle('Average cell number, first 40 timesteps vs Time to progression')
    # # one legend for all 3 axes middle right
    # fig.legend(loc='center right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

