import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot(df, title, scale='linear', truncate=False):
    fig, ax = plt.subplots()
    if truncate:
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        truncated = df[((df['Type 0'] + df['Type 1'])/initial_size >= 1.3)]
        print(truncated)
        if len(truncated) > 0:
            index = truncated.index[0]
            # replace df with zeros after index
            df.loc[index:, 'Type 0'] = 0
            df.loc[index:, 'Type 1'] = 0
            df.loc[index:, 'Treatment'] = 0
    ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
    ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
    ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
    ax.legend()
    ax.set_title(title)
    ax.set_yscale(scale)
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    treat = np.where(treat == 0, np.roll(treat, 1), treat)
    ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
    lw=2)
    return ax

def get_ttps(filename, timesteps=100):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        # find the largest index with non-zero Type 0 and Type 1
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        nz = df[((df['Type 0'] + df['Type 1'])/initial_size > 1.33)]
        if len(nz) > 0:
            # append index when type 0 + type 1 is larger than 1.5
            ttps.append(nz.index[0])
        else:
            ttps.append(len(df))

    return ttps

def get_mutant_proportions(filename, timesteps=100):
    mutant_proportions = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key=f'run_{i}')
        initial_size = df['Type 0'][0] + df['Type 1'][0]
        # mutant_proportions.append(df['Type 1'].values[-1]/(df['Type 0'].values[-1] + df['Type 1'].values[-1]))
        nz = df[((df['Type 0'] + df['Type 1']) / initial_size > 1.33)]
        if len(nz) > 0:
            mutant_proportions.append(nz['Type 1'].values[0] / (nz['Type 0'].values[0] + nz['Type 1'].values[0]))
        else:
            mutant_proportions.append(df['Type 1'].values[-1] / (df['Type 0'].values[-1] + df['Type 1'].values[-1]))
    return mutant_proportions

def main():
    PC_files_list = [
                    # 'data/2D_benchmarks/no_treatment/2d_no_treatment_all.h5',
                    #  'data/2D_benchmarks/mtd/2d_mtd_all.h5',
                    #  'data/2D_benchmarks/at50/2d_at50_all.h5',
                    #  'data/2D_benchmarks/at100/2d_at100_all.h5',
                    #  'data/2D_benchmarks/fixed_1_1/2d_fixed_1_1_all.h5',
                    #  'data/2D_benchmarks/fixed_1_2/2d_fixed_1_2_all.h5',
                    #  'data/2D_benchmarks/fixed_1_25/2d_fixed_1_25_all.h5',
                     'data/2D_benchmarks/x6/2d_x6_all.h5',
                     'data/2D_benchmarks/n2_t4_l3/2d_n2_t4_l3_all.h5',
                     'data/2D_benchmarks/s2_t5_l3/2d_s2_t5_l3_all.h5',
                     # 'data/2D_benchmarks/n1_t4/2d_n1_t4_all.h5',
                     # 'data/2D_benchmarks/s1_t4/2d_s1_t4_all.h5',
                     # 'data/2D_benchmarks/x7_t6/2d_x7_t6_all.h5',
                     # 'data/2D_benchmarks/x8_t3/2d_x8_t3_all.h5',
                     #
                     #'data/2D_benchmarks/slvenv_agent_1/2d_slvenv_agent_1_all.h5',
                     #'data/2D_benchmarks/x8_t2/2d_x8_t2_all.h5',
                     # './data/2D_benchmarks/interm_slvenv_t2/2d_interm_slvenv_t2_all.h5',
                     # './data/2D_benchmarks/slv_t3/2d_slv_t3_all.h5',
                     #'./Evaluations/LvEnvEval_ref72803_x7_noise_refine_7_punish20.h5',
                     #'data/2D_benchmarks/random/2d_random_all.h5'
                     ]
    PC_name_list = [
        #'PC No therapy', 'PC MTD', 'PC AT50', 'PC AT100', 'PC fixed 1.1', 'PC fixed 1.2',
                    'PC x6', 'PC n2', 'PC s2',
                    #'PC fixed 1.25', 'PC x6', 'PC n1', 'PC s1', 'PC rand'
    ]

    PC_dict = {}
    mut_prop_dict = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])
        mut_prop_dict[PC_name_list[i]] = get_mutant_proportions(PC_files_list[i])

    # classify the mutant proportions into 3 categories, low, medium and high
    classified_mut_prop = {}
    for i in range(len(PC_name_list)):
        classified_mut_prop[PC_name_list[i]] = np.zeros(len(mut_prop_dict[PC_name_list[i]]))
        for j in range(len(mut_prop_dict[PC_name_list[i]])):
            if mut_prop_dict[PC_name_list[i]][j] < 0.01 and PC_dict[PC_name_list[i]][j] < 251:
                classified_mut_prop[PC_name_list[i]][j] = 0
            elif mut_prop_dict[PC_name_list[i]][j] < 0.5 and PC_dict[PC_name_list[i]][j] < 251:
                classified_mut_prop[PC_name_list[i]][j] = 1
            elif mut_prop_dict[PC_name_list[i]][j] >= 0.5 and PC_dict[PC_name_list[i]][j] < 251:
                classified_mut_prop[PC_name_list[i]][j] = 2
            else:
                classified_mut_prop[PC_name_list[i]][j] = 3

    # pie chart classification

    for i in range(len(PC_name_list)):
        fig, ax = plt.subplots()
        ax.pie([np.sum(classified_mut_prop[PC_name_list[i]] == 0),
                np.sum(classified_mut_prop[PC_name_list[i]] == 1),
                np.sum(classified_mut_prop[PC_name_list[i]] == 2),
                np.sum(classified_mut_prop[PC_name_list[i]] == 3)],
               labels=['premature progression', 'resistance rise', 'resistance > 50%', 'resistance contained'], autopct='%1.1f%%', startangle=90)
        ax.set_title(PC_name_list[i])


    PC_df = pd.DataFrame(PC_dict)
    mut_prop_df = pd.DataFrame(mut_prop_dict)

    # plot the mutant proportions against pc
    fig, ax = plt.subplots()
    for i in range(len(PC_name_list)):
        ax.scatter(mut_prop_df[PC_name_list[i]], PC_df[PC_name_list[i]], label=PC_name_list[i])
    # ax.scatter(mut_prop_df[PC_name_list[0]], PC_df[PC_name_list[0]], label=PC_name_list[0])
    ax.set_title('Mutant proportions against PC')
    ax.legend()

    LV_files_list = ['./Evaluations/LvEnvEval_2d_no_treatment.h5',
                        './Evaluations/LvEnvEval_2d_mtd.h5',
                        './Evaluations/LvEnvEval_2d_at50.h5',
                        './Evaluations/LvEnvEval_2d_at100.h5',
                        './Evaluations/LvEnvEval_2d_fixed_1_1.h5',
                        './Evaluations/LvEnvEval_2d_fixed_1_2.h5',
                        './Evaluations/LvEnvEval__test_corrected_noise_lv.h5',
                        './Evaluations/LvEnvEval_greatest_agent_run2703_test_x6.h5',
                        './data/newer_trains_savings/Evaluations/LvEnvEval_nn1104_n1_t4.h5',
                        './data/newer_trains_savings/Evaluations/LvEnvEval_nn1104_s1_t4.h5',
                        # './Evaluations/LvEnvEval_x8_t2_l5_on_lvnoise2803_x8_cobra_t2_5.h5',
                        #
                        #'./Evaluations/LvEnvEval_x8_t7_l3_on_lvnoise3103_x8_cobra_t7_load_3.h5',
                        #'./Evaluations/LvEnvEval_x8_t2_l5_on_lvnoise2803_x8_cobra_t2_5.h5',
                        './Evaluations/LvEnvEval_2d_random.h5'

                        ]
    LV_name_list = ['LV No therapy', 'LV MTD', 'LV AT50', 'LV AT100', 'LV fixed 1.1', 'LV fixed 1.2',
                    #'LV x6 RL', 'LV x8 t7',
                    'LV fixed 1.25', 'LV x6 agent', 'LV n1', 'LV s1', 'LV rand']

    LV_dict = {}
    for i in range(len(LV_files_list)):
        LV_dict[LV_name_list[i]] = get_ttps(LV_files_list[i])

    LV_df = pd.DataFrame(LV_dict)

    SLV_files_list = ['./Evaluations/SLvEnvEval__-no_treatment.h5',
                        './Evaluations/SLvEnvEval__-mtd.h5',
                        './Evaluations/SLvEnvEval__-at50.h5',
                        './Evaluations/SLvEnvEval__-at100.h5',
                        './Evaluations/SLvEnvEval__-fixed_1_1.h5',
                        './Evaluations/SLvEnvEval__-fixed_1_2.h5',
                        #'./Evaluations/SLvEnvEval__slvenv_agent_on_itself_20304_slvenv_train_try.h5',
                        './Evaluations/SLvEnvEval__-fixed_1_25.h5',
                        './Evaluations/SLvEnvEval_agent_slv0804_quasi_fitted_slvenv_train_try_load_2.h5',
                        './Evaluations/SLvEnvEval__-random.h5'
                        ]
    SLV_name_list = ['SLV No therapy', 'SLV MTD', 'SLV AT50', 'SLV AT100', 'SLV fixed 1.1',
                     'SLV fixed 1.2', 'SLV fixed 1.25', 'SLV agent on itself', 'SLV rand']

    SLV_dict = {}
    for i in range(len(SLV_files_list)):
        SLV_dict[SLV_name_list[i]] = get_ttps(SLV_files_list[i])

    SLV_df = pd.DataFrame(SLV_dict)

    # combine the two dataframes
    combined = {}
    for i in range(len(PC_name_list)):
        combined[PC_name_list[i]] = PC_df[PC_name_list[i]]
        #combined[LV_name_list[i]] = LV_df[LV_name_list[i]]
        # combined[SLV_name_list[i]] = SLV_df[SLV_name_list[i]]
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax, width = 0.3)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')

    return combined_df


combined_df = main()
plt.show()
