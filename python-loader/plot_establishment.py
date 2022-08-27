import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

# matplotlib font 
import matplotlib 
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 10,
                    'font.weight': 'normal',
                    'font.family': 'serif'})
#plt.style.use('default')
#plt.style.use('dark_background')

class Establishment:
    """ This class is for plotting establishment probabilities """ 
    
    def __init__(self,filename,key): 
        self.df = pd.read_hdf(filename,key)
        self.mutation_rate = 1e-1
            
    def survival_probability(self): 
        """
        Survival probability conditioned on that you switched 
        """
        return self.df['survivals']/self.df['tot_switched'] 
    def mutation_probability(self):
        """
        Probability that clone of the average width would get at least one mutation: 1 - Prob(no cell switches) = 1 - Prob(a cell does not switch) ^ ( number of cells ) = 1 - (1-\mu)^(N)
        
        """
        return 1 - (1-self.mutation_rate)**(self.df['red_cells']/self.df['tot_switched'])
        
    def average_size(self):
        """
        Return average clone size at the point of mutation
        """
        return self.df['red_cells']/self.df['tot_switched']
    def mutate_and_survive_probability(self): 
        """
        Probability that you will mutate and also survive
        """
        return ( self.df['survivals']/self.df['tot_switched'] ) * ( 1 - (1-self.mutation_rate)**(self.df['red_cells']/self.df['tot_switched']) )

def error(a,b): 
    """
    error of a/b for Poisson
    """
    return a/b*(np.sqrt( (np.sqrt(a)/a)**2 + (np.sqrt(b)/b)**2))

list_of_keys = ['data_cobra5','data_cobra10','data_cobra15','data_cobra20','data_raven30','data_raven35','data_raven40','data_raven_other145','data_raven50']#,,'data_raven_other220']
list_of_switching_times = [5,10,15,20,30,35,40,45,50]
survival_prb = []
mutation_prb = []
combined_prb = []

filename = r'data\survivals.h5'; 


for key, time in zip(list_of_keys,list_of_keys) : 
    
    Et = Establishment(filename,key) 
    survival_prb.append(Et.survival_probability())
    mutation_prb.append(Et.mutation_probability())
    combined_prb.append(Et.mutate_and_survive_probability())
    #plot_survival(ax1,df,'k') 
    #plot_width(ax2,df,'k') 
    #plot_combined(ax3,df,'k')

fig, axs = plt.subplots()
axs.plot(list_of_switching_times,survival_prb)
ax2 = axs.twinx()
axs.plot(list_of_switching_times,mutation_prb,color='r')
ax2.plot(list_of_switching_times,combined_prb,color='c')
plt.show()
"""
f = h5py.File(filename, 'r')

print(list(f.keys()))


def plot_survival(ax,df,color):
   
    
    tot_switched = df['tot_switched']
    survivals = df['survivals'] 
    red_cells = df['red_cells']
    time = df['switching_time']
    
    ax.errorbar(time, survivals/tot_switched, error(survivals,tot_switched),capsize=4, ls='', color = color, marker = 'o', linewidth = 2) 
    #plt.ylim([0,1])
    ax.set_yticks([.200,.400,.600])
    ax.set_yticklabels([.200,.400,.600],rotation=90)
    ax.set_xlabel(r'$R^{*}$')
    ax.xaxis.set_label_coords(1., -0.025)
    ax.set_ylabel(r'$P_{est}$',rotation=0)
    ax.yaxis.set_label_coords(0.05, 1.0)
    #ax.set_title('Establishment probability')
    #plt.tight_layout()
    return 0 
    
def plot_width(ax,df,color):
   
    
    tot_switched = df['tot_switched']
    survivals = df['survivals'] 
    red_cells = df['red_cells']
    time = df['switching_time']
    
    ax.scatter(time,red_cells/tot_switched,color = color ) 
    #plt.ylim([0,1])
    #ax.set_yticks([200,400,600])
    #ax.set_yticklabels([200,400,600],rotation=90)
    ax.set_xlabel(r'$R^{*}$')
    ax.xaxis.set_label_coords(1., -0.025)
    ax.set_ylabel(r'$N_{cells}$',rotation=0)
    ax.yaxis.set_label_coords(0.05, 1.0)
    #ax.set_title('Cell number')
    #plt.tight_layout()
    return 0 

def plot_combined(ax,df,color):
  
    
    tot_switched = df['tot_switched']
    survivals = df['survivals'] 
    red_cells = df['red_cells']
    time = df['switching_time']
    
    ax.errorbar(time, red_cells*survivals/tot_switched, red_cells*error(survivals,tot_switched),capsize=4, ls='', color = color, marker = 'o', linewidth = 2) 
    #plt.ylim([0,1])
    ax.set_xlim(xmin=0,xmax=56)
    ax.set_xlabel(r'$R^{*}$')
    ax.xaxis.set_label_coords(1., -0.025)
    ax.set_ylabel(r'$P_{full}$',rotation=0)
    ax.yaxis.set_label_coords(0.05, 1.0)
    #ax.set_title('Combined probability')
    #plt.tight_layout()
    return 0     

cm = 1/2.54
fig1, ax1 = plt.subplots(figsize=(5*cm,5*cm))
fig2, ax2 = plt.subplots(figsize=(5*cm,5*cm))
fig3, ax3 = plt.subplots(figsize=(5*cm,5*cm))

for key in ['data_cobra5','data_cobra10','data_cobra15','data_cobra20']: 
    df = pd.read_hdf(filename, key)
    print(df)
    plot_survival(ax1,df,'k') 
    plot_width(ax2,df,'k') 
    plot_combined(ax3,df,'k')

for key in ['data_raven30','data_raven35','data_raven40','data_raven50']: 
    df = pd.read_hdf(filename, key)
    plot_survival(ax1,df,'k')
    plot_width(ax2,df,'k') 
    plot_combined(ax3,df,'k')
    
for key in ['data_raven_other145','data_raven_other220']: 
    df = pd.read_hdf(filename, key)
    plot_survival(ax1,df,'k')
    plot_width(ax2,df,'k') 
    plot_combined(ax3,df,'k')
#fig1.savefig('images\Survival.svg',transparent=True)

#fig2.savefig('images\Cell_num.svg',transparent=True)

#fig3.savefig('images\Combined.svg',transparent=True)
plt.show()
"""
