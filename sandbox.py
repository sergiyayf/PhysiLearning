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

# Loop through every key in the file
ttps_no_treat = get_ttps('Evaluations/PcEnvEvalpatient_80_no_treatment.h5')
ttps_mtd = get_ttps('Evaluations/PcEnvEvalpatient_80_mtd.h5')
ttps_random = get_ttps('Evaluations/PcEnvEvalpatient_80_random.h5')
ttps_at = get_ttps('Evaluations/PcEnvEvalpatient_80_AT_at_baseline.h5')
ttps_LV_at = get_ttps('Evaluations/LvEnvEvalpatient_80_lv_fixed_cap_test.h5')
ttps_LV_no_treat = get_ttps('Evaluations/LvEnvEvalLV-no_treat.h5')
ttps_LV_mtd = get_ttps('Evaluations/LvEnvEvalLV-mtd.h5')
ttps_LV_random = get_ttps('Evaluations/LvEnvEvalLV-random.h5')
ttps_PC_rl = get_ttps('Evaluations/PcEnvEvalpatient_80_RL_training_images_1407.h5')

df = pd.DataFrame({'MTD': ttps_mtd,
                   'LV_MTD': ttps_LV_mtd,
                   'AT': ttps_at,
                   'RL treat PC': ttps_PC_rl,
                   'LV_AT': ttps_LV_at,
                   'Random': ttps_random,
                   'LV_Random': ttps_LV_random,
                   } )

# box plot the distribution with scatter using seaborn
fig, ax = plt.subplots()
sns.boxplot(data=df, ax=ax)
sns.stripplot(data=df, ax=ax, color='black', jitter=0.2, size=2.5)
# show mean as well
ax.scatter(df.mean().index, df.mean().values, marker='o', color='red', s=20)

# plot one trajectory of aT scenario
df = pd.read_hdf('Evaluations/LvEnvEvalLV-mtd.h5', key='run_0')
fig, ax = plt.subplots()
ax.plot(df.index, df['Type 0'], label='Type 0')
ax.plot(df.index, df['Type 1'], label='Type 1')
ax.plot(df.index, df['Type 0']+df['Type 1'], label='total')
ax.legend()
ax.fill_between(df.index, df['Treatment']*4000, df['Treatment']*4250, color='orange', label='drug', lw=0)


plt.show()
