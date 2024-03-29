import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams.update({'font.size': 20,
                         'font.weight': 'normal',
                         'font.family': 'sans-serif'})
mpl.rcParams['pdf.fonttype'] = 42  # to make text editable in pdf output
#mpl.rcParams['font.sans-serif'] = ['Arial']  # to make it Arial

def plot_trajectory(ax,df,episode=0):

    #df = pd.read_csv(f'../../../data/070223_raven_model_tests/Evaluations/{episode}_070223_minelikeparams_rewf=4_growthf=0.csv', index_col=[0])
    #df = pd.read_csv(f'../../../data/070223_raven_model_tests/Evaluations/0_070223_JKlikeparams_rewf=4_growthf=0_fixedAT.csv', index_col=[0])
    x = np.arange(0,len(df))/24
    ax.fill_between(x, df['Treatment']*3000, df['Treatment']*3250, color='orange', label='drug', lw=0)
    ax.plot(x, (df['Type 0'] + df['Type 1']), 'k', label='total', linewidth=2)
    ax.plot(x, df['Type 0'], 'b', label='wt', linewidth=2)
    ax.plot(x, df['Type 1'], 'g', label='mut', linewidth=2)
    #ax.set_xlabel('time')
    #ax.set_ylabel('# Cells')
    #ax.set_title(f'PC_evaluation')


fig, ax = plt.subplots(1,5,figsize=(16, 6))
fig.suptitle('LV_Zhang')
for i in range(0,5):
    df = pd.read_csv(
        f'/home/saif/Projects/PhysiLearning/Evaluations/{i}_LV_Zhang_test.csv',
        index_col=[0])
    plot_trajectory(ax[i],df)

fig, ax = plt.subplots(1,5,figsize=(16, 6))
fig.suptitle('PC_Zhang')
for i in range(0,5):
    df = pd.read_csv(
        f'/home/saif/Projects/PhysiLearning/Evaluations/{i}_PC_Zhang_test.csv',
        index_col=[0])
    plot_trajectory(ax[i],df)

fig, ax = plt.subplots(1,5,figsize=(16, 6))
fig.suptitle('LV_policy')
for i in range(0,5):
    df = pd.read_csv(
        f'/home/saif/Projects/PhysiLearning/Evaluations/{i}_LVEval09_03_PPO_with_3comp_new_LV_rew0.csv',
        index_col=[0])
    plot_trajectory(ax[i],df)
#
# for i in range(10,14):
#     df = pd.read_csv(
#         f'/home/saif/Projects/PhysiLearning/{i-5}_Eval_Zhang85_LV_test.csv',
#         index_col=[0])
#     plot_trajectory(ax[i//5,i%5],df)
#
# i = 15
# df = pd.read_csv(
#     f'/home/saif/Projects/PhysiLearning/0_Eval__test.csv',
#     index_col=[0])
# plot_trajectory(ax[i//5,i%5],df)
# fig, ax = plt.subplots(figsize=(16, 8))
# df = pd.read_csv(
#         f'/home/saif/Projects/PhysiLearning/newLV_test_for_longer_LSTM_0_AT_p1.csv',
#         index_col=[0])
# plot_trajectory(ax,df)

#fig.savefig(f'/home/saif/Projects/PhysiLearning/data/images/PC_eval.svg',transparent=True)
plt.show()