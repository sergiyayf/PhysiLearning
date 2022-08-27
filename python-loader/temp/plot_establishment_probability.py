import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24,
                    'font.weight': 'normal'})
#plt.style.use('default')
plt.style.use('dark_background')

time_when_switched = [5,10,15,20,30,35,40,50]
averaged_probability = [0.567,0.4842,0.4154,0.3545,0.3997,0.3835,0.3151,0.24795]
averaged_std = [0.064, 0.14887,0.093,0.0944,0.06747,0.1552,0.10967,0.11963]
plt.figure(figsize=(10,10))
(_, caps, _) = plt.errorbar(time_when_switched, averaged_probability, averaged_std,capsize=20, ls='', color = 'orange', marker = 'o', linewidth = 4, ms = 10, mfs= None) 
plt.ylim([0,1])
#plt.tight_layout()
plt.xlabel('Time of rescue')
plt.ylabel('Survival probability')


for cap in caps:
    
    cap.set_markeredgewidth(4)

plt.savefig('establishment.png')
#plt.show()
