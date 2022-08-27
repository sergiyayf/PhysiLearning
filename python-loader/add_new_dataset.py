from load.h5tohist import *
import pandas as pd
from auxiliary.analyze_history import *
# add data to the files

filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211129_sw_at_35\29_11_sw_at_35_run_1\output\all_data.h5'
col_name = 'sw_35_diff_9'
h5File = 'growth_layer.h5'
write_to_new_collection(col_name,filename,h5File = h5File)

for i in range(1,11):
    filename = r'D:\Serhii\Projects\EvoChan\Simulations\sim_results_garching\Important_Selection_escape_sims\20211129_sw_at_35\29_11_sw_at_35_run_'+str(i)+r'\output\all_data.h5'
    write_to_existing_collection(col_name,filename,h5File = h5File)


