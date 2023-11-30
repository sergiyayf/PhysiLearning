import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def get_ttps(filename, timesteps=70):
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
                        'Evaluations/older_evals/lv_on_pc_2108_combined.h5',
                        'Evaluations/older_evals/PcEnvEval_pc_rl_1008_interruption_combined.h5',
                        'Evaluations/PcEnvEval_20231027_transfer.h5',
                        'data/results/eval_transfered_r0/PcEnvEval__transfer_r020231102_PC_transfer_r0_changed_callback_1.h5',
                        'Evaluations/PcEnvEval_20231103_transfer_r0to8.h5',
                        'data/results/eval_lv_r0/PcEnvEval__patient_93_no_treatment20231031_patient_80_add_noise_2.h5',
                        'data/results/eval_lv_r8/PcEnvEval__r8_lv_trained20231102_patient_80_r8_with_noise.h5',
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
                    'PC AT50', 'PC AT75', 'PC AT100', 'PC Random',
                    'PC RL(lv trained)', 'PC RL', 'Transfer','tr r0', 'tr t8',
                    'r0', 'r8']

    # PC_files_list = patient_4_files_list
    # PC_name_lsit = patient_4_name_list

    LV_files_list = ['Evaluations/Lv/LvEnvEval_patient_80_no_treatment.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_mtd.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT50.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT75.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT100.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_random.h5',
                        'Evaluations/older_evals/LvEnvEvallv_2108_cont_2.h5',
                        'Evaluations/LvEnvEval_lv_noise20231106_patient_80_r8_with_noise_load.h5',
                     ]
    LV_name_lsit = ['LV No treatment', 'LV MTD',
                    'LV AT50', 'LV AT75', 'LV AT100', 'LV Random', 'LV RL', 'newest lv noise' ]

    PC_dict = {}
    LV_dict = {}
    combined = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_lsit[i]] = get_ttps(PC_files_list[i])

    for i in range(len(LV_files_list)):
        LV_dict[LV_name_lsit[i]] = get_ttps(LV_files_list[i])

    for i in range(len(PC_files_list)):
        combined[PC_name_lsit[i]] = get_ttps(PC_files_list[i])
        # combined[LV_name_lsit[i]] = get_ttps(LV_files_list[i])

    PC_df = pd.DataFrame(PC_dict)
    LV_df = pd.DataFrame(LV_dict)
    # combined_df = pd.DataFrame(combined)

    # box plot the distribution with scatter using seaborn
    fig, ax = plt.subplots()
    sns.boxplot(data=PC_df, ax=ax, width = 0.3)
    sns.stripplot(data=PC_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(PC_df.mean().index, PC_df.mean(), marker='x', color='red', s=50, label='mean')

    fig, ax = plt.subplots()
    sns.boxplot(data=LV_df, ax=ax, width = 0.5)
    sns.stripplot(data=LV_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # show mean as well
    ax.scatter(LV_df.mean().index, LV_df.mean(), marker='x', color='red', s=50, label='mean')

    # fig, ax = plt.subplots()
    # sns.boxplot(data=combined_df, ax=ax)
    # sns.stripplot(data=combined_df, ax=ax, color='black', jitter=0.2, size=2.5)
    # # show mean as well
    # ax.scatter(combined_df.mean().index, combined_df.mean(), marker='x', color='red', s=50, label='mean')
    # #fig.savefig('all_treatments.pdf', transparent=True)

    for i in range(1, 0):
        df = pd.read_hdf('Evaluations/LvEnvEval_lv_noise20231106_patient_80_r8_with_noise_load.h5', key=f'run_{i}')
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Type 0'], label='Type 0')
        ax.plot(df.index, df['Type 1'], label='Type 1')
        ax.plot(df.index, df['Type 0'] + df['Type 1'], label='total')
        ax.legend()
        ax.set_title(f'Patient 80, lv r0 run 5')
        # ax.set_yscale('log')
        ax.fill_between(df.index, df['Treatment'] * 4000, df['Treatment'] * 4250, color='orange', label='drug', lw=0)

    for j in [1, 4, 55, 80, 93]:
        for i in range(1, 2):
            df = pd.read_hdf(f'Evaluations/LvEnvEval__pat_{j}_20231108_rew12_cohort.h5', key=f'run_{i}')
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Type 0'], label='Type 0')
            ax.plot(df.index, df['Type 1'], label='Type 1')
            ax.plot(df.index, df['Type 0'] + df['Type 1'], label='total')
            ax.legend()
            ax.set_title(f'eval r8 pat{j} run {i}')
            # ax.set_yscale('log')
            ax.fill_between(df.index, df['Treatment'] * 4000, df['Treatment'] * 4250, color='orange', label='drug',
                            lw=0)

    for j in [4, 55, 80, 93]:
        for i in range(1, 2):
            df = pd.read_hdf(f'Evaluations/LvEnvEval__pat_{j}_20231106_patient_80_r8_with_noise_load.h5',
                             key=f'run_{i}')
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Type 0'], label='Type 0')
            ax.plot(df.index, df['Type 1'], label='Type 1')
            ax.plot(df.index, df['Type 0'] + df['Type 1'], label='total')
            ax.legend()
            ax.set_title(f'only p80 pat{j} run {i}')
            # ax.set_yscale('log')
            ax.fill_between(df.index, df['Treatment'] * 4000, df['Treatment'] * 4250, color='orange', label='drug',
                            lw=0)

    for j in [80]:
        for i in range(1, 2):
            df = pd.read_hdf(f'Evaluations/LvEnvEval_test_rew020231024_patient_80_retraining_r0.h5', key=f'run_{i}')
            fig, ax = plt.subplots()
            ax.plot(df.index, df['Type 0'], label='Type 0')
            ax.plot(df.index, df['Type 1'], label='Type 1')
            ax.plot(df.index, df['Type 0'] + df['Type 1'], label='total')
            ax.legend()
            ax.set_title(f'only p80 pat{j} run {i}')
            # ax.set_yscale('log')
            ax.fill_between(df.index, df['Treatment'] * 4000, df['Treatment'] * 4250, color='orange', label='drug',
                            lw=0)

fig0, ax0 = plt.subplots()
df = pd.read_hdf(f'Evaluations/LvEnvEvalon_off.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax0.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1, 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'1 on off')
# ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)
print("1 average: ",df['Type 0'].mean())

df = pd.read_hdf(f'Evaluations/LvEnvEvalon_off_double.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax0.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1, 2')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'2 on off')
# ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)
print("2 average: ",df['Type 0'].mean())

df = pd.read_hdf(f'Evaluations/LvEnvEvalon_off_triple.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax0.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1, 3')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'3 on off')
# ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)
print("3 average: ",df['Type 0'].mean())

ax0.legend()

for i in range(1,6):
    df = pd.read_hdf(f'data/degeneracy_tests/LV_deg_{i}/Evaluations/LvEnvEval_policy_{i}test_v0.3.0b_conda_env.h5', key=f'run_0')
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
    ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
    ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
    ax.legend()
    ax.set_title(f'Policy {i}')
    # ax.set_yscale('log')
    treat = df['Treatment'].values
    # replace 0s that are directly after 1 with 1s
    #treat = np.where(treat == 0, np.roll(treat, 1), treat)
    ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
    lw=2)
    print("3 average: ",df['Type 0'].mean())
# main()
plt.show()
