import pandas as pd
import seaborn as sns
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot( df, title, scale='linear', truncate=False, ax=None, c='black'):
    if ax is None:
        fig, ax = plt.subplots()
    if truncate:
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.4)]
        print(truncated)
        index = truncated.index[0]
        # replace df with zeros after index
        df.loc[index:, 'Type 0'] = 0
        df.loc[index:, 'Type 1'] = 0
        df.loc[index:, 'Treatment'] = 0

    skip = 6
    normalize = 2
    sus = np.array(df['Type 0'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    res = np.array(df['Type 1'].values[::skip] )#/ (df['Type 0'][normalize] + df['Type 1'][normalize]))
    # sus = np.array(df['Type 0'].values)*70
    # res = np.array(df['Type 1'].values)*70
    tot = sus + res
    time = df.index[::skip] / skip
    # only do for nonzero tot
    time = time[tot > 0]
    sus = sus[tot > 0]
    res = res[tot > 0]
    tot = tot[tot > 0]

    treatment_color = '#A8DADC'
    color = '#EBAA42'
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('# cells (norm)')
    ax.plot(time, res, color=color, marker='x',
                label='Resistant', markersize=4)
    # ax.plot(data['Day'], data['Red'], color=color)
    ax.tick_params(axis='y')

    color = '#5E82B8'
    ax.plot(time, sus, color=color, marker='x',
            label='Sensitive', markersize=4)

    ax.plot(time, tot, color='k', marker='x',
            label='Total', markersize=4)

    # ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    ax.hlines(1.0, 0, time[-1], color='k', linestyle='--')
    treat = np.array(df['Treatment'].values[::skip])
    treat = treat[:len(time)]
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, -1), treat)
    for t in range(len(time)-1):
        if treat[t] == 1:
            ax.axvspan((t-1), t, color=treatment_color)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cell count')
    # ax.legend()
    return ax

def get_ttps(filename, timesteps=10):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 1'])/initial_size >= 1.0)]
        tot_idx = df[((df['Type 0'] + df['Type 1'])/initial_size >= 2.0)].index
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0]/2)
        elif len(tot_idx) > 0:
            ttps.append(tot_idx[0]/2)
        else:
            ttps.append(len(df)/2)
        #ttps.append(df['Type 1'][52]/initial_size)
    return ttps

def main():
    PC_files_list = ['./Evaluations/physicell_0901_tain_6_data/run_1/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_1.h5',
                        './Evaluations/physicell_0901_tain_6_data/run_2/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_2.h5',
                        './Evaluations/physicell_0901_tain_6_data/run_3/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_3.h5',
                        './Evaluations/physicell_0901_tain_6_data/run_4/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_4.h5',
                        './Evaluations/physicell_0901_tain_6_data/run_5/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_5.h5',
                     ]
    PC_name_list = ['pc 1', 'pc 2', 'pc 3', 'pc 4', 'pc 5']

    PC_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    PC_df = pd.DataFrame(PC_dict)

    LV_files_list = ['Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_1.h5',
                        'Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_2.h5',
                        'Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_3.h5',
                        'Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_4.h5',
                        'Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_5.h5',

                        ]
    LV_name_list = ['lv 1', 'lv 2', 'lv 3', 'lv 4', 'lv 5']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]

    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')
    ax.set_ylim(0, 300)

def plot_trajectories():

    fig, axs = plt.subplots(5,10, figsize=(20, 5))
    for i in range(1,11):
        for j in range(5):
            # fig, ax = plt.subplots()

            #df = pd.read_hdf(f'./Evaluations/physicell_0901_tain_6_data/run_{i}/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_{i}.h5', key=f'run_{j}')
            #df = pd.read_hdf(f'./Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5', key=f'run_{j}')
            #df = pd.read_hdf(f'./Evaluations/slvenv_1901_train_average_2m/SLvEnvEval__agnt_20250116_SLV_train_{i}.h5', key=f'run_{j}'
            #df = pd.read_hdf(f'./Evaluations/slv_2301_hypertune_2/SLvEnvEval__agnt_20250123_slv_hypers_2.h5',
            #                 key=f'run_{j}')
            df = pd.read_hdf(f'./Evaluations/23012025_slv_train/SLvEnvEval__agnt_20250123_slv_run_{i}.h5', key=f'run_{j}')
            ax = axs[j,i-1]
            plot(df, f'PC Daily', scale='linear', truncate=False, ax = ax, c='red')
            # calculate average totatl cell count
            tot = df['Type 0'] + df['Type 1']
            tot = tot[tot > 0]
            print(np.mean(tot))
            max_sens = np.max(df['Type 0'])
            min_sens = np.min(df['Type 0'])
            min_tot = np.min(tot)
            ax.set_title(f'C: {np.mean(tot):.2f}, A: {min_tot:.2f}')
            ax.set_xlim(0, 100)
            ax.set_ylim(0,2)
            #ax.set_yscale('log')
            #fig.savefig(f'./plots/lucky_4_pc_{i}.pdf')


def plot_cell_number_distribution():
    i = 4
    j = 1
    ls = []
    avgt = []
    performance = []
    maxis = []
    for i in range(1,11):
        sens = []
        total = []
        ttps = []

        for j in range(10):
            # df = pd.read_hdf(
            #     f'./Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5',
            #     key=f'run_{j}')
            df = pd.read_hdf(
                f'./Evaluations/physicell_0901_tain_6_data/run_{i}/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_{i}.h5',
                key=f'run_{j}')
            sensitive_cells = df['Type 0'][0:100]
            lowest_sensitive = np.min(sensitive_cells)
            tot = df['Type 0']# + df['Type 1']
            tot = tot[tot > 0]
            average_total = np.mean(tot)
            total.append(average_total)
            sens.append(lowest_sensitive)
            initial_size = df['Type 0'][0] + df['Type 1'][0]
            nz = df[((df['Type 1']) / initial_size >= 1.0)]
            tot_idx = df[((df['Type 0'] + df['Type 1']) / initial_size >= 2.0)].index
            if len(nz) > 0:
                # append index when type 0 + type 1 is larger than 1.5
                ttps.append(nz.index[0] / 2)
            elif len(tot_idx) > 0:
                ttps.append(tot_idx[0] / 2)
            else:
                ttps.append(len(df) / 2)
            y = np.array(sensitive_cells)
            maxima = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
            # A point is a minimum if it's less than both neighbors
            minima = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
            # Get the indices of maxima and minima
            maxima_indices = np.where(maxima)[0] + 1  # Adjust index due to slicing
            minima_indices = np.where(minima)[0] + 1  # Adjust index due to slicing

        ls.append(sens)
        avgt.append(total)
        performance.append(ttps)
    print(ls)
    print(avgt)
    print(performance)
    ls = np.array(ls).flatten()
    avgt = np.array(avgt).flatten()
    performance = np.array(performance).flatten()
    fig, ax = plt.subplots()
    # plot performance against avgt
    ax.scatter(ls, performance)
    ax.set_xlabel('Lowest sensitive cell count')
    ax.set_ylabel('Time to progression')
    fig, ax = plt.subplots()
    # plot performance against avgt
    ax.scatter(avgt, performance)
    ax.set_xlabel('Average total cell count')
    ax.set_ylabel('Time to progression')

    # take dta with average total cell count > 0.93
    indices = np.where(avgt > 0.93)
    ls = ls[indices]
    avgt = avgt[indices]
    performance = performance[indices]
    fig, ax = plt.subplots()
    # plot performance against avgt
    ax.scatter(ls, performance)
    ax.set_xlabel('Lowest sensitive cell count')
    ax.set_ylabel('Time to progression')
    ax.set_title('Average total cell count > 0.93')

    fig, ax = plt.subplots()
    ax.scatter(maxis, performance)
    ax.set_xlabel('Maxima')

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
    average_total = np.mean(tot[:40])
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
    for file in list_of_files:
        ttps = []
        avgs = []
        for run in range(10):
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
        file_idx += 1
    return data_dict, averaged_data



if __name__ == '__main__':

    #main()
    plot_trajectories()
    #plot_cell_number_distribution()
    list_of_files_pc = [
        f'./Evaluations/physicell_0901_tain_6_data/run_{i}/PcEnvEval_run20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
        for i in range(1, 11)]
    data_dict_pc, average_data_pc = analyze_data(list_of_files_pc)
    list_of_files_lv = [f'./Evaluations/0901_onehalf_day_6/LvEnvEval__agnt_20250109_2DLV_average_less_1_onehalf_day_{i}.h5'
                     for i in range(1, 11)]
    data_dict_lv, average_data_lv = analyze_data(list_of_files_lv)
    list_of_files_slv = [f'./Evaluations/slvenv_1901_train_average_2m/SLvEnvEval__agnt_20250116_SLV_train_{i}.h5'
                        for i in range(1, 11)]
    data_dict_slv, average_data_slv = analyze_data(list_of_files_slv)
    # scatter avg vs ttp, low vs ttp, high vs tpp and low_max vs ttp

    fig2, ax2 = plt.subplots()
    for key in average_data_lv.keys():
        ax2.scatter(average_data_lv[key]['AVG'], average_data_lv[key]['TTP'], label=f'LV {key+1}')
    #ax2.scatter([average_data[key]['AVG'] for key in average_data.keys()], [average_data[key]['TTP'] for key in average_data.keys()])
    ax2.legend()
    ax2.set_title('LV predictor')

    fig2, ax2 = plt.subplots()
    for key in average_data_slv.keys():
        ax2.scatter(average_data_slv[key]['AVG'], average_data_slv[key]['TTP'], label=f'LV {key+1}')
    # ax2.scatter([average_data[key]['AVG'] for key in average_data.keys()], [average_data[key]['TTP'] for key in average_data.keys()])
    ax2.legend()
    ax2.set_title('SLV-LV predictor')
    plt.show()

