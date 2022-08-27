import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import matplotlib

def main():
    h5File = 'growth_layer.h5'
    f = h5py.File(h5File, 'r')
    fig, ax = plt.subplots()
    for diffusion in [7,9,11]:
        growth_layer = pd.read_hdf(h5File,'data/sw_35_diff_'+str(diffusion)+'/ds1/growth_layer')
        ax.plot(growth_layer)
    plt.show()
    return 0

if __name__ == '__main__':
    main()
