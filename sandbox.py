import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def get_ttps(filename, timesteps=100):
    ttps = []
    for i in range(timesteps):
        df = pd.read_hdf(filename, key='run_'+str(i))
        # find the largest index with non-zero Type 0 and Type 1
        nz = df[(df['Type 0'] + df['Type 1'] != 0)]
        if len(nz) > 0:
            ttps.append(nz.index[-1])
        else:
            ttps.append(0)
    return ttps

def main():
    PC_files_list = ['Evaluations/Pc/PcEnvEval_patient_80_no_treatment.h5',
                        'Evaluations/Pc/PcEnvEval_patient_80_mtd.h5',
                        'Evaluations/Pc/PcEnvEval_patient_80_AT50.h5',
                        'Evaluations/Pc/PcEnvEval_patient_80_AT75.h5',
                        'Evaluations/Pc/PcEnvEval_patient_80_AT100.h5',
                        'Evaluations/Pc/PcEnvEval_patient_80_random.h5',
                        'Evaluations/lv_on_pc_2108_combined.h5',
                        'Evaluations/PcEnvEval_pc_rl_1008_interruption_combined.h5',
                     ]
    pat_x = 93
    patient_4_files_list = [f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_no_treatment.h5',
                                f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_mtd.h5',
                                f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_AT50.h5',
                                f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_AT75.h5',
                                f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_AT100.h5',
                                f'data/training_patients_benchmarks/patient_{pat_x}_results/patient_{pat_x}_random.h5',
                                ]
    patient_4_name_list = [f'Patient {pat_x} No treatment', f'Patient {pat_x} MTD',
                    f'Patient {pat_x} AT50', f'Patient {pat_x} AT75', f'Patient {pat_x} AT100', f'Patient {pat_x} Random']

    PC_name_lsit = ['PC No treatment', 'PC MTD',
                    'PC AT50', 'PC AT75', 'PC AT100', 'PC Random', 'PC RL(lv trained)', 'PC RL']

    # PC_files_list = patient_4_files_list
    # PC_name_lsit = patient_4_name_list

    LV_files_list = ['Evaluations/Lv/LvEnvEval_patient_80_no_treatment.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_mtd.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT50.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT75.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT100.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_random.h5',
                        'Evaluations/LvEnvEvallv_2108_cont_2.h5',
                        'Evaluations/LvEnvEvallv_2108_cont_2.h5',
                     ]
    LV_name_lsit = ['LV No treatment', 'LV MTD',
                    'LV AT50', 'LV AT75', 'LV AT100', 'LV Random', 'LV RL', 'LV RL']

    PC_dict = {}
    LV_dict = {}
    combined = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_lsit[i]] = get_ttps(PC_files_list[i])

    for i in range(len(LV_files_list)):
        LV_dict[LV_name_lsit[i]] = get_ttps(LV_files_list[i])

    for i in range(len(PC_files_list)):
        combined[PC_name_lsit[i]] = get_ttps(PC_files_list[i])
        combined[LV_name_lsit[i]] = get_ttps(LV_files_list[i])

    PC_df = pd.DataFrame(PC_dict)
    LV_df = pd.DataFrame(LV_dict)
    combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=PC_df, ax=ax)
    sns.stripplot(data=PC_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(PC_df.mean().index, PC_df.mean(), marker='x', color='red', s=50, label='mean')

    fig, ax = plt.subplots()
    sns.boxplot(data=LV_df, ax=ax)
    sns.stripplot(data=LV_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(LV_df.mean().index, LV_df.mean(), marker='x', color='red', s=50, label='mean')

    fig, ax = plt.subplots()
    sns.boxplot(data=combined_df, ax=ax)
    sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')
    #fig.savefig('all_treatments.pdf', transparent=True)

# plot one trajectory of aT scenario
#
df = pd.read_hdf('Evaluations/LvEnvEval20_09_LV_multiobs_patient_sampling.h5', key='run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'], label='Type 0')
ax.plot(df.index, df['Type 1'], label='Type 1')
ax.plot(df.index, df['Type 0']+df['Type 1'], label='total')
ax.legend()
ax.fill_between(df.index, df['Treatment']*4000, df['Treatment']*4250, color='orange', label='drug', lw=0)

df = pd.read_hdf('Evaluations/LvEnvEvallv_rl_p80_test_at100.h5', key='run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'], label='Type 0')
ax.plot(df.index, df['Type 1'], label='Type 1')
ax.plot(df.index, df['Type 0']+df['Type 1'], label='total')
ax.legend()
ax.fill_between(df.index, df['Treatment']*4000, df['Treatment']*4250, color='orange', label='drug', lw=0)

df = pd.read_hdf('Evaluations/LvEnvEval20_09_LV_multiobs_patient_sampling_patient_1.h5', key='run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'], label='Type 0')
ax.plot(df.index, df['Type 1'], label='Type 1')
ax.plot(df.index, df['Type 0']+df['Type 1'], label='total')
ax.legend()
ax.fill_between(df.index, df['Treatment']*4000, df['Treatment']*4250, color='orange', label='drug', lw=0)

main()
plt.show()
