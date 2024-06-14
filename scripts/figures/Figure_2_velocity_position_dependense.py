import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

os.chdir('/home/saif/Projects/PhysiLearning')
average_projections = pd.read_pickle('data/pickles/mean_velocity_projections.pkl')
std_proj = pd.read_pickle('data/pickles/std_velocity_projections.pkl')
bin_size = 20
fig, ax = plt.subplots()

rads = [x*bin_size for x in range(0,len(average_projections))]
rads = np.array(rads)
ax.errorbar(rads, average_projections, yerr=std_proj, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
ax.set_xlabel('Distance to front')
ax.set_ylabel('Normal velocity magnitude')
ax.set_title('Instant radial velocity')

ax.axhline(0, color='black', lw=1)

# fit linear to the first 5 points
def linear(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear, rads[:4], average_projections.values[:4], sigma=std_proj[:4])
print(popt)

def trunc_linear(x, a, b):
    return (a * x + b)*np.heaviside(a*x+b, 1)

ax.plot(rads, trunc_linear(rads, *popt), 'g-', label='fit 5 points')
ax.legend()
plt.show()