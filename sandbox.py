import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def get_ttps(filename, timesteps=40):
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

# Loop through every key in the file
ttps_no_treat = get_ttps('Evaluations/PcEnvEvalpatient_80_no_treatment.h5')
ttps_mtd = get_ttps('Evaluations/PcEnvEvalpatient_80_mtd.h5')
ttps_random = get_ttps('Evaluations/PcEnvEvalpatient_80_random.h5')
ttps_at100 = get_ttps('Evaluations/PcEnvEvalpatient_80_AT_at_baseline.h5')
ttps_at50 = get_ttps('Evaluations/PcEnvEvalpatient_80_AT50.h5')
ttps_at75 = get_ttps('Evaluations/PcEnvEvalpatient_80_AT75.h5')
ttps_LV_at = get_ttps('Evaluations/LvEnvEvalpatient_80_lv_fixed_cap_test.h5')
ttps_LV_no_treat = get_ttps('Evaluations/LvEnvEvalLV-no_treat.h5')
ttps_LV_mtd = get_ttps('Evaluations/LvEnvEvalLV-mtd.h5')
ttps_LV_random = get_ttps('Evaluations/LvEnvEvalLV-random.h5')
ttps_PC_rl = get_ttps('Evaluations/PcEnvEvalpatient_80_RL_training_images_1407.h5')
ttps_PC_rl_new = get_ttps('Evaluations/PcEnvEvalpatient_80_RL_new_1407.h5')

ttps_agent = get_ttps('Evaluations/PcEnvEval1008_interruption_with_57.h5')

df = pd.DataFrame({'MTD': ttps_mtd,
                   'LV_MTD': ttps_LV_mtd,
                   'AT100': ttps_at100,
                   'LV_AT': ttps_LV_at,
                   'Random': ttps_random,
                   'LV_Random': ttps_LV_random,
                   'RL treat PC': ttps_PC_rl_new,
                   } )

df_ats = pd.DataFrame({'AT50': ttps_at50,
                       'AT75': ttps_at75,
                       'AT100': ttps_at100,
                       'agent': ttps_agent,
                       'LV_AT100': ttps_LV_at,
                       } )

# box plot the distribution with scatter using seaborn
fig, ax = plt.subplots()
sns.boxplot(data=df, ax=ax)
sns.stripplot(data=df, ax=ax, color='black', jitter=0.2, size=2.5)
ax.scatter(df.mean().index, df.mean().values, marker='o', color='red', s=20)

# box plot the distribution with scatter using seaborn
fig, ax = plt.subplots()
sns.boxplot(data=df_ats, ax=ax)
sns.stripplot(data=df_ats, ax=ax, color='black', jitter=0.2, size=2.5)
ax.scatter(df_ats.mean().index, df_ats.mean().values, marker='o', color='red', s=20)

# show mean as well


# plot one trajectory of aT scenario
df = pd.read_hdf('Evaluations/PcEnvEvalpatient_80_AT50.h5', key='run_20')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'], label='Type 0')
ax.plot(df.index, df['Type 1'], label='Type 1')
ax.plot(df.index, df['Type 0']+df['Type 1'], label='total')
ax.legend()
ax.fill_between(df.index, df['Treatment']*4000, df['Treatment']*4250, color='orange', label='drug', lw=0)


plt.show()
