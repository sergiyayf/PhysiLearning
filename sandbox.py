import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def get_ttps(filename, timesteps=45):
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
    PC_files_list = [
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_80_f_1_2.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_80_f_1_3.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_93_f_0_8.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_93_f_0_9.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_93_f_1_0.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_279_f_0_6.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_279_f_0_7.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_279_f_0_8.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_94_f_0_6.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_94_f_0_7.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_4_f_0_4.h5',
        'Evaluations/critical_threshold/PcEnvEvalfixed_pat_4_f_0_5.h5',

                        #'Evaluations/Pc/PcEnvEval_patient_80_no_treatment.h5',
                        #'Evaluations/Pc/PcEnvEval_patient_80_mtd.h5',
                        #'Evaluations/Pc/PcEnvEval_patient_80_AT50.h5',
                        #'Evaluations/Pc/PcEnvEval_patient_80_AT75.h5',
                        #'Evaluations/Pc/PcEnvEval_patient_80_AT100.h5',
                        #'Evaluations/Pc/PcEnvEval_patient_80_random.h5',
                        #'Evaluations/older_evals/lv_on_pc_2108_combined.h5',
                        #'Evaluations/older_evals/PcEnvEval_pc_rl_1008_interruption_combined.h5',
                        #'Evaluations/PcEnvEval_20231027_transfer.h5',
                        #'data/results/eval_transfered_r0/PcEnvEval__transfer_r020231102_PC_transfer_r0_changed_callback_1.h5',
                        #'Evaluations/PcEnvEval_20231103_transfer_r0to8.h5',
                        #'data/results/eval_lv_r0/PcEnvEval__patient_93_no_treatment20231031_patient_80_add_noise_2.h5',
                        #'data/results/eval_lv_r8/PcEnvEval__r8_lv_trained20231102_patient_80_r8_with_noise.h5',
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

    PC_name_list = ['89 f 1.2', '89 f 1.3', '135 f 0.8', '135 f 0.9', '135 f 1.0', '170 f 0.6',
                    '170 f 0.7', '170 f 0.8', '210 f 0.6', '210 f 0.7', '295 f 0.4', '295 f 0.5']

    # PC_files_list = patient_4_files_list
    # PC_name_lsit = patient_4_name_list

    LV_files_list = ['Evaluations/Lv/LvEnvEval_patient_80_no_treatment.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_mtd.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT50.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT75.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_AT100.h5',
                        'Evaluations/Lv/LvEnvEval_patient_80_random.h5',
                        'Evaluations/older_evals/LvEnvEvallv_2108_cont_2.h5',
                        #'Evaluations/LvEnvEval_lv_noise20231106_patient_80_r8_with_noise_load.h5',
                     ]
    LV_name_lsit = ['LV No treatment', 'LV MTD',
                    'LV AT50', 'LV AT75', 'LV AT100', 'LV Random', 'LV RL' ]

    PC_dict = {}
    LV_dict = {}
    combined = {}
    for i in range(len(PC_files_list)):
        PC_dict[PC_name_list[i]] = get_ttps(PC_files_list[i])

    for i in range(len(LV_files_list)):
        LV_dict[LV_name_lsit[i]] = get_ttps(LV_files_list[i])

    for i in range(len(PC_files_list)):
        combined[PC_files_list[i]] = get_ttps(PC_files_list[i])
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




df = pd.read_hdf(f'Evaluations/SLvEnvEval_testSLV_test_2024.h5', key=f'run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
ax.legend()
ax.set_title(f'AT65 Spatial LV')
# ax.set_yscale('log')
treat = df['Treatment'].values
# replace 0s that are directly after 1 with 1s
#treat = np.where(treat == 0, np.roll(treat, 1), treat)
ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
lw=2)
#
# for i in [5,9,10]:
#     df = pd.read_hdf(f'Evaluations/Pc/PcEnvEval_patient_80_AT100.h5', key=f'run_{i}')
#     fig, ax = plt.subplots()
#     ax.plot(df.index, df['Type 0'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 0')
#     ax.plot(df.index, df['Type 1'].values/(df['Type 0'][0]+df['Type 1'][0]), label='Type 1')
#     ax.plot(df.index, (df['Type 0'] + df['Type 1'])/(df['Type 0'][0]+df['Type 1'][0]), label='total')
#     ax.legend()
#     ax.set_title(f'AT100 PC')
#     ax.set_yscale('log')
#     treat = df['Treatment'].values
#     # replace 0s that are directly after 1 with 1s
#     #treat = np.where(treat == 0, np.roll(treat, 1), treat)
#     ax.fill_between(df.index, 1, 1.250, where=treat==1, color='orange', label='drug',
#     lw=2)

#main()
plt.show()
