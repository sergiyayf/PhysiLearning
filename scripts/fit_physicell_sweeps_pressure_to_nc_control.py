import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py


with h5py.File('./../data/physicell_arrest_pressure_change_to_fit_nc/PcEnvEval_job_13440464nt3_1.h5', 'r') as f:
    print(f.keys())
    keys = list(f.keys())

fig, ax = plt.subplots()
for key in keys:
    df = pd.read_hdf('./../data/physicell_arrest_pressure_change_to_fit_nc/PcEnvEval_job_13440464nt3_1.h5', key=key)
    ax.plot(df.index, df['Type 0'], label=key)

ax.legend()
plt.show()