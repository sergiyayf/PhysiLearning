import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.colors as mcolors
ex = {'font.size': 6,
          'font.weight': 'normal',
          'pdf.fonttype': 42,
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial',
          'errorbar.capsize': 2,
          }
plt.rcParams.update(ex)
K = 18000/3589
r_s = 0.04
r_r = 0.134
c = 4.78
d_s = 0.01*r_s
d_r = 0.01*r_r

def color_plot(df, max_time, ax=None, colormap = plt.cm.viridis):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the number of resistat cells vs total, color the libe with index of the timepoint
    tot = df['Type 0'] + df['Type 1']
    res = df['Type 1']
    res = res[tot > 0]
    sus = df['Type 0']
    sus = sus[tot > 0]
    tot = tot[tot > 0]

    for i in range(max_time):
        ax.plot(sus[i:i+2], res[i:i+2], color=colormap(i/max_time))
    #ax.axhline(y=np.mean(tot), color='k', linestyle='--')
    #ax.set_xscale('log')
    # plot colormap
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max_time/4))
    sm.set_array([])
    #plt.colorbar(sm, ax=ax, label='Time')
    #ax.set_xlabel('Sensitive cells')
    #ax.set_ylabel('Resistant cells')
    return ax

def ds_dt(S, R):
    return r_s * S *(1 - (R + S) / K) - d_s *S

def dr_dt(S, R):
    return r_r * R *(1 - (R + c * S) / K) - d_r*R

if __name__ == '__main__':
    os.chdir('/home/saif/Projects/PhysiLearning')


    # create a grid of S and R values
    S = np.linspace(0, 2.05, 150)
    R = np.linspace(0, 2.05, 150)
    S, R = np.meshgrid(S, R)
    # compute the time derivative of S and R
    dS_dt = ds_dt(S, R)
    dR_dt = dr_dt(S, R)
    # plot the vectors

    fig, ax = plt.subplots(figsize=(150 / 72, 150 / 72), constrained_layout=True)
    vmax = np.max(abs(dR_dt))*0.9
    vmin = -vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    #norm = mcolors.SymLogNorm(linthresh=0.1, linscale=0.5, vmin=-vmax, vmax=vmax, base=10)
    contourf = ax.contourf(S, R, dR_dt, levels=0, cmap='RdBu', norm=norm, alpha=0.5)
    #quiver = ax.quiver(S, R, dS_dt, dR_dt)
    # plot stream lines
    ax.streamplot(S, R, dS_dt, dR_dt, color='k')
    ax.set_xlabel('Sensitive cells')
    ax.set_ylabel('Resistant cells')

    # equal axes
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 2.05)
    fig.savefig(f'./scripts/figures/plots/Figure_2_separatrix.pdf', transparent=True)

    ######## Fig 2
    # create a grid of S and R values
    S = np.linspace(0.5, 1.5, 10)
    R = np.linspace(0, 0.075, 10)
    S, R = np.meshgrid(S, R)
    # compute the time derivative of S and R
    dS_dt = ds_dt(S, R)
    dR_dt = dr_dt(S, R)


    fig2, ax2 = plt.subplots(figsize=(150 / 72, 150 / 72), constrained_layout=True)
    vmax = np.max(abs(dR_dt))
    vmin = -vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    contourf = ax2.contourf(S, R, dR_dt, levels=0, cmap='RdBu', norm=norm, alpha=0.5)
    #quiver = ax2.quiver(S, R, dS_dt, dR_dt)
    ax2.streamplot(S, R, dS_dt, dR_dt, color='k', density=0.5)
    i=9
    j=2
    df = pd.read_hdf(f'./Evaluations/1402_lvs_evals/LvEnvEval__20250206_lv_1_{i}.h5', key=f'run_{j}')
    #df = pd.read_hdf(f'./Evaluations/1402_pcs_evals/run_{i}.h5', key=f'run_{j}')
    colormap = plt.cm.Oranges_r
    color_plot(df[:600], 720, ax2, colormap)
    ax2.set_ylim(0, 0.075)
    ax2.set_xlim(0.5, 1.5)
    #color_plot(df[:], 720, ax, colormap)
    # remove all x and y ticks and labels
    # edit ticks to scientific notation
    #ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    ax2.set_xlabel('Sensitive cells ')
    ax2.set_ylabel('Resistant cells')
    ax.set_aspect('equal', 'box')
    fig2.savefig(f'./scripts/figures/plots/Figure_2_separatrix_zoomin.pdf', transparent=True)

    # plot colormap as figure 3 independently
    # fig3, ax3 = plt.subplots(figsize=(100 / 72, 150 / 72), constrained_layout=True)
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=720 / 4))
    # sm.set_array([])
    # plt.colorbar(sm, ax=ax3, label='Time')
    #fig3.savefig(f'./scripts/figures/plots/Figure_2_colormap.pdf', transparent=True)

    plt.show()
