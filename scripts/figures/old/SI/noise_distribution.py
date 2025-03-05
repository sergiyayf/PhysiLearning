import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, truncnorm

type_0 = []
for time in range(1,6):
    for i in range(100):
        df = pd.read_hdf('/home/saif/Projects/PhysiLearning/data/2D_benchmarks/no_treatment/2d_no_treatment_all.h5', key=f'run_{i}')
        # df = pd.read_hdf('/home/saif/Projects/PhysiLearning/data/2D_benchmarks/at100/2d_at100_all.h5', key=f'run_{i}')
        type_0.append((df['Type 0'].values[time]-df['Type 0'].values[time-1])/df['Type 0'].values[time-1])

# plot the distribution
tpye_0 = np.array(type_0)
fig, ax = plt.subplots()
sns.histplot(type_0-np.mean(type_0), kde=True, label = 'Data', color = 'blue', ax=ax)
plt.show()

# approximate the distribution with a normal distribution
mean = 0
std = np.std(type_0)
x = np.linspace(-0.025, 0.025, 100)
y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
ax.plot(x, y, label='Normal distribution', color = 'red')

# plot truncated normal distribution, that is zero for values below -0.02 and above 0.02
y = truncnorm.pdf(x, -0.02/std, 0.02/std, loc=mean, scale=std)

ax.plot(x, y, label='Truncated normal distribution', color = 'green')

ax.legend()
ax.set_title('Distribution of noise')
ax.set_xlabel('Noise strength (relative to previous time step)')
ax.set_ylabel('Frequency')

fig, ax = plt.subplots()
# plot normal distribution
x = np.linspace(-0.05, 0.05, 100)
mean = 0
#std = 0.01
y = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))

ax.plot(x, y, label='Normal distribution', color = 'red')
ax.legend()
plt.show()
