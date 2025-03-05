import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
ex = {'font.size': 8,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 3,
          }
plt.rcParams.update(ex)

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


def get_largest_sensitive(df):
    sensitive_cells = df['Type 0']
    largest_sensitive = np.max(sensitive_cells)
    return largest_sensitive

def get_lowest_maximum(df):
    y = np.array(df['Type 0']+df['Type 1'])
    maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    maxima_indices = np.where(maxima)[0] + 1  # Adjust index due to slicing
    m = y[maxima_indices]
    # find the maxima_indices with the lowest value
    min_value = np.min(m)
    min_index = maxima_indices[np.where(m == min_value)]
    return min_value

def get_index_of_min_max(df):
    y = np.array(df['Type 0'] + df['Type 1'])
    maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    maxima_indices = np.where(maxima)[0] + 1  # Adjust index due to slicing
    m = y[maxima_indices]
    return np.argmin(m)

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

            #print(f'Run {run} TTP: {ttp}, AVG: {avg}, LOW: {low}, HIGH: {high}, LOW_MAX: {low_max}')
            data_dict[f'{file_idx}_{run}'] = {'TTP': ttp, 'AVG': avg, 'MIN_MIN': min_min, 'MAX_MIN': max_min,
                                              'AVG_MIN': avg_min, 'MIN_MAX': min_max, 'MAX_MAX': max_max, 'AVG_MAX': avg_max}

            ttps.append(ttp)
            avgs.append(avg)
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
    #data_dict_pc = data_dict_lv

    avg_max = [data_dict_pc[key]['AVG_MAX'] for key in data_dict_pc.keys()]
    avg_min = [data_dict_pc[key]['AVG_MIN'] for key in data_dict_pc.keys()]
    avg = [data_dict_pc[key]['AVG'] for key in data_dict_pc.keys()]
    min_min = [data_dict_pc[key]['MIN_MIN'] for key in data_dict_pc.keys()]
    min_max = [data_dict_pc[key]['MIN_MAX'] for key in data_dict_pc.keys()]
    max_min = [data_dict_pc[key]['MAX_MIN'] for key in data_dict_pc.keys()]
    max_max = [data_dict_pc[key]['MAX_MAX'] for key in data_dict_pc.keys()]
    ttp = [data_dict_pc[key]['TTP'] for key in data_dict_pc.keys()]

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # shuffle the indices
    indices = np.arange(len(avg_max))
    np.random.shuffle(indices)
    avg_max = np.array(avg_max)[indices]
    avg_min = np.array(avg_min)[indices]
    avg = np.array(avg)[indices]
    min_min = np.array(min_min)[indices]
    min_max = np.array(min_max)[indices]
    max_min = np.array(max_min)[indices]
    max_max = np.array(max_max)[indices]
    ttp = np.array(ttp)[indices]

    X = np.array([avg_max, avg_min, avg, min_min, min_max, max_min, max_max]).T
    y = np.array(ttp)
    feature_names = ['avg_max', 'avg_min', 'avg', 'min_min', 'min_max', 'max_min', 'max_max']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    # Display the importances in DF
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)


    plt.show()

