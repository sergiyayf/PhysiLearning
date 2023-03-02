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
    x = np.arange(0,len(df))
    ax.fill_between(x, 0, df['Treatment']*1, color='orange', label='drug', lw=0)
    ax.plot(x, (df['Type 0'] + df['Type 1']), 'k', label='total', linewidth=2)
    ax.plot(x, df['Type 0'], 'b', label='wt', linewidth=2)
    ax.plot(x, df['Type 1'], 'g', label='mut', linewidth=2)
    #ax.set_xlabel('time')
    #ax.set_ylabel('# Cells')
    #ax.set_title(f'PC_evaluation')


# fig, ax = plt.subplots(2,4,figsize=(8, 4))
# for i in range(1,5):
#     df = pd.read_csv(
#         f'/home/saif/Projects/PhysiLearning/Full_0_AT_p{i}.csv',
#         index_col=[0])[0:500]
#     plot_trajectory(ax[0,i-1],df,i)
    # df2 = pd.read_csv(
    #     f'/home/saif/Projects/PhysiLearning/Zhang_AT_0_AT_p{i}.csv',
    #     index_col=[0])[0:500]
    # plot_trajectory(ax[1,i-1],df2,i)
    #plot_trajectory(ax[i//2,i%2],df2,i)

fig, ax = plt.subplots(figsize=(16, 8))
df = pd.read_csv(
        f'/home/saif/Projects/PhysiLearning/newLV_test_for_longer_LSTM_0_AT_p1.csv',
        index_col=[0])
plot_trajectory(ax,df)

#fig.savefig(f'/home/saif/Projects/PhysiLearning/data/images/PC_eval.svg',transparent=True)
plt.show()