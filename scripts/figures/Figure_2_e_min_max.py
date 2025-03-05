import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)

def get_ttp(df):
    initial_size = df['Type 0'][0] + df['Type 1'][0]
    nz = df[((df['Type 1']) / initial_size >= 1.0)]
    tot_idx = df[((df['Type 0'] + df['Type 1']) / initial_size >= 2.0)].index
    if len(nz) > 0:
        # append index when type 0 + type 1 is larger than 1.5
        return nz.index[0] / 4
    elif len(tot_idx) > 0:
        return tot_idx[0] / 4
    else:
        return len(df) / 4

def get_average_cell_number(df, type='Total'):
    if type == 'Total':
        tot = df['Type 0'] + df['Type 1']
    else:
        tot = df[type]
    tot = tot[tot > 0]
    average_total = np.mean(tot[:])
    return average_total

def get_prm(df, type='min', min_max='min'):
    y = np.array(df['Type 0'] + df['Type 1'])
    y = y[y > 0]
    minima = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    minima_indices = np.where(minima)[0] + 1  # Adjust index due to slicing
    m = y[minima_indices]
    maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    maxima_indices = np.where(maxima)[0] + 1  # Adjust index due to slicing
    mm = y[maxima_indices]

    if type == 'min':
        if min_max == 'min':
            return np.min(m)
        elif min_max == 'max':
            return np.min(mm)
    elif type == 'max':
        if min_max == 'min':
            return np.max(m)
        elif min_max == 'max':
            return np.max(mm)
    elif type == 'avg':
        if min_max == 'min':
            return np.mean(m)
        elif min_max == 'max':
            return np.mean(mm)

def get_argminmax(df):
    y = np.array(df['Type 0'] + df['Type 1'])
    treatment = np.array(df['Treatment'])
    # get values at treatment on
    y = y[treatment > 0]
    indices = np.where(treatment > 0)
    min_treated = np.min(y)
    argmintreated = np.argmin(y)
    index = indices[0][argmintreated]
    return index

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
        low_maxs=[]
        for run in range(50):
            df = pd.read_hdf(file, key=f'run_{run}')
            ttp = get_ttp(df)
            avg = get_average_cell_number(df)
            min_min = get_prm(df, 'min', 'min')
            max_min = get_prm(df, 'max', 'min')
            avg_min = get_prm(df, 'avg', 'min')
            min_max = get_prm(df, 'min', 'max')
            max_max = get_prm(df, 'max', 'max')
            avg_max = get_prm(df, 'avg', 'max')
            time_min_treated = get_argminmax(df)
            #print(f'Run {run} TTP: {ttp}, AVG: {avg}, LOW: {low}, HIGH: {high}, LOW_MAX: {low_max}')
            data_dict[f'{file_idx}_{run}'] = {'TTP': ttp, 'AVG': avg, 'MIN_MIN': min_min, 'MAX_MIN': max_min,
                                              'AVG_MIN': avg_min, 'MIN_MAX': min_max, 'MAX_MAX': max_max, 'AVG_MAX': avg_max,
                                              'TIME_MIN_TREATED': time_min_treated}

            ttps.append(ttp)
            avgs.append(min_max)
        averaged_data[file_idx] = {'TTP': np.mean(ttps), 'AVG': np.mean(avgs)}
        std_data[file_idx] = {'TTP': np.std(ttps), 'AVG': np.std(avgs)}
        file_idx += 1
    return data_dict, averaged_data, std_data



if __name__ == '__main__':
    os.chdir('/home/saif/Projects/PhysiLearning')
    data = 'new'
    if data == 'old':
        list_of_files_pc = [
            f'./Evaluations/train_6_physicell_evals/train_6_run_{i}.h5'
            for i in range(1, 11)]
        list_of_files_lv = [f'./Evaluations/train_6_lv_evals/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
                         for i in range(1, 11)]
    elif data == 'new':
        list_of_files_pc = [
            f'./Evaluations/1402_pcs_evals/run_{i}.h5'
            for i in range(1, 11)]
        list_of_files_lv = [
            f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5'
            for i in range(1, 11)]
    data_dict_lv, average_data_lv, std_data_lv = analyze_data(list_of_files_lv)
    data_dict_pc, average_data_pc, std_data_pc = analyze_data(list_of_files_pc)
    # scatter avg vs ttp, low vs ttp, high vs tpp and low_max vs ttp
    #fig, ax = plt.subplots( figsize=(250 / 72, 150 / 72), constrained_layout=True)
    fig, ax = plt.subplots(figsize=(130 / 72, 130 / 72), constrained_layout=True)
    for key in average_data_lv.keys():
        # scatter plot with error bars (average_data_lv[key]['AVG'], average_data_lv[key]['TTP'], label=f'LV {key+1}')
        ax.errorbar(average_data_lv[key]['AVG'], average_data_lv[key]['TTP'],
                       xerr=std_data_lv[key]['AVG'], yerr=std_data_lv[key]['TTP'], fmt='o', markersize=3, color='grey')

    pearson_corr = np.corrcoef([average_data_lv[key]['AVG'] for key in average_data_lv.keys()], [average_data_lv[key]['TTP'] for key in average_data_lv.keys()])
    print('LV Pearson:', pearson_corr)

    for key in average_data_lv.keys():
        ax.errorbar(average_data_pc[key]['AVG'], average_data_pc[key]['TTP'],
                     xerr=std_data_pc[key]['AVG'], yerr=std_data_pc[key]['TTP'], fmt='o', markersize=3, color='darkgrey')

    # calculate pearsons correlation
    pearson_corr = np.corrcoef([average_data_pc[key]['AVG'] for key in average_data_pc.keys()], [average_data_pc[key]['TTP'] for key in average_data_pc.keys()])
    print('PC Pearson:', pearson_corr)

    #ax.set_title('PC')
    #ax[1].set_xlim(0.25, 0.65)
    ax.set_ylim(0, 220)

    # set one x label for all 3 axes
    ax.set_xlabel('Lowest treatment oon')
    ax.set_ylabel('TTF')
    #fig.suptitle('Average cell number, first 40 timesteps vs Time to progression')
    # one legend for all 3 axes middle right
    fig.savefig('./scripts/figures/plots/Figure_2_predictors_min_max_new.pdf', transparent=True)

    # check for general correlation
    x = [data_dict_pc[key]['TIME_MIN_TREATED'] for key in data_dict_pc.keys()]
    y = [data_dict_pc[key]['TTP'] for key in data_dict_pc.keys()]
    pearson_corr = np.corrcoef(x, y)
    print('PC general Pearson:', pearson_corr)
    plt.show()

