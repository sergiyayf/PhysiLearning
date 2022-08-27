from load.h5tohist import *
import pandas as pd
from auxiliary.analyze_history import *
from matplotlib import pyplot as plt 
from cycler import cycler

def fit_line(x,y,weights): 
    new_x = x[~np.isnan(y)]
    new_y = y[~np.isnan(y)]
    new_w = weights[~np.isnan(y)]
   
    coef, cov = np.polyfit(new_x,new_y,1, cov = 'unscaled') 
    
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    #slope = 1000*coef[0]
    #ax.plot(new_x, poly1d_fn(new_x), '--k')
    #ax.text(new_x[-1],poly1d_fn(new_x[-1]),'%.1f'% slope, color = 'k')
   
    return coef, cov; 

h5File = 'ER_data_collection.h5'
f = h5py.File(h5File, 'r')
print(f['data'].keys())
print(f['data/tailored_switch_at_45/ds5'].attrs['filename'])


df = pd.read_hdf(h5File, 'data/tailored_switch_at_50/ds3/red')

df2 = pd.read_hdf(h5File, 'data/tailored_switch_at_50/ds3/blue')
width = [df,df2]

cm = 1/2.54
fig, ax = plt.subplots(figsize=(5*cm,12*cm))
# plots the history 
plot_sorted(ax,sort(width), width)

# stuff for tau escape 
ncolors=9
fig, ax = plt.subplots(figsize=(10*cm,8*cm))
#set colors 
ax.set_prop_cycle(cycler('color', plt.get_cmap('viridis', ncolors).colors) *
                   cycler('linestyle', ['-', '--']))

# loop through all the tailored switching samples 
max_width_dead_had = []
max_width_alive_had = [] 
rhos_escape = []
r_star = []
rhos_escape_err = []
slopes = []
for switch in [5,15,20,25,30,35,40,45,50]:
    r_w = []
    b_w = []
    t_w = [] 
    b_w_std = [] 
    fulldf = pd.DataFrame()
    bluedf = pd.DataFrame() 
    for key in f['data/tailored_switch_at_'+str(switch)]:
        
        # get red, blue and total width 
        df = pd.read_hdf(h5File, 'data/tailored_switch_at_'+str(switch)+'/'+key+'/red')
        df2 = pd.read_hdf(h5File, 'data/tailored_switch_at_'+str(switch)+'/'+key+'/blue')
        tot = df.add(df2, axis=1,fill_value=0)
        
        # condition on survival
        condition_on_survival = False
        if condition_on_survival == True: 
            print('this is wrong')
            d = df2.tail(1).iloc[0].isna()
            d = pd.DataFrame(d)
            
            select_indices = list(np.where(d[d.keys()[0]] == False)[0])
            
            dd = d.iloc[select_indices]
            
            key_names = dd.index.values.tolist()
            new_df2 = df2[key_names]
            df2 = new_df2
        
        # remove rad column and reindex to concatenate all the streaks of this condition
        temptot = tot.reset_index()
        temptot = temptot.drop(columns=['rad'])
        
        tempblue = df2.reset_index()
        tempblue = tempblue.drop(columns=['rad'])
        
        #print(temptot)
        fulldf = pd.concat([fulldf,temptot],axis = 1) 
        bluedf = pd.concat([bluedf,tempblue],axis = 1)
               
        # get the max width of those who died and did not 
        for k in df2.keys(): 
            vals = df2[k].values
            
            if np.isnan(vals[-1]):
                max_width_dead_had.append(np.nanmax(vals))
                #print(vals)
                #print('maxv',np.nanmax(vals))
            else: 
                max_width_alive_had.append(np.nanmax(vals))
    
    # average for this condition 
    rad = df.index
    average_blue_w = bluedf.mean(axis=1) 
    std_blue_w = bluedf.std(axis=1) 
    #ax.errorbar(rad, average_blue_w,yerr = std_blue_w, label = switch)
    ax.plot(rad, average_blue_w, label = switch)
    weights = 1/std_blue_w
    
    coef, cov = fit_line(rad,average_blue_w,weights)
    print('coefs', coef)
    print('covariance', cov)
    err = np.sqrt(np.diag(cov))
    print('slope', coef[0],' pm ',err[0])
    print('slope', coef[1],' pm ',err[1])
    poly1d_fn = np.poly1d(coef) 
    # larger x 
    x = rad[~np.isnan(average_blue_w)]
    x = np.linspace(x[0], 9000, 1000)
    ax.plot(x,poly1d_fn(x),'k--')
    
    # prepare confidence level curves
    nstd = 1. # to draw 5-sigma intervals
    popt_up = coef + nstd * err
    popt_dw = coef - nstd * err
    
    func = np.poly1d(coef)
    fit = func(x)
    fit_up = np.poly1d(popt_up)
    
    fit_dw = np.poly1d(popt_dw)
    
    ax.fill_between(x, fit_up(x), fit_dw(x), alpha=.25)
    # get rho escape 
    escape_width = 7 
    x_large = min(x[poly1d_fn(x)>escape_width])
    x_large_up = min(x[fit_up(x)>escape_width])
    #x_large_dw = min(x[fit_dw(x)>escape_width])
    
    rho_escape = x_large - x[0] 
    rho_escape_err = x_large_up-x_large
    # get rho escape
    rhos_escape.append(rho_escape)
    rhos_escape_err.append(rho_escape_err)
    r_star.append(x[0])
    slopes.append(coef[0])


ax.axhline(y=np.nanmax(max_width_dead_had),xmin=0,xmax=1,ls='--', color = 'k', label = 'escape') 
ax.axhline(y=6,xmin=0,xmax=1,ls='--', color = 'k') 
ax.axhline(y=7,xmin=0,xmax=1,ls='--', color = 'k') 
ax.legend()     
print(max_width_dead_had)
print('dead width = ', np.nanmax(max_width_dead_had))
print(max_width_alive_had)
print('alive width = ', np.nanmax(max_width_alive_had))

print('len tot = ', len(max_width_alive_had)+len(max_width_dead_had))

print('unique widths of dead = ',np.unique(max_width_dead_had,return_counts = True )) 

unique, counts = np.unique(max_width_dead_had,return_counts = True )
for un, c in zip(unique,counts):
    print('Probability to die with width ',un)
    print(c/(len(max_width_alive_had)+len(max_width_dead_had)))
    

fig, ax = plt.subplots(figsize=(10*cm,10*cm))
ax.errorbar(r_star,rhos_escape,yerr = rhos_escape_err, color = 'k')
ax.set_xlabel(r'$\rho^{*}$')
ax.set_ylabel(r'$\rho_{escape}$')
fig, ax = plt.subplots(figsize=(10*cm,10*cm))
ax.scatter(slopes,rhos_escape)
ax.legend()
plt.show()
