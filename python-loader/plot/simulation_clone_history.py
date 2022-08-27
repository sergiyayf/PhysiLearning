import h5py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from auxiliary import get_perifery

def plot_history(filename):
    history_read = pd.read_hdf(filename,'data/history');
    history_read = history_read.fillna(value=0.)
    #print(history_read)
    color_history_read = pd.read_hdf(filename,'data/color_history');
    color_mask_type_1 = color_history_read.copy(); 
    color_mask_type_1[color_mask_type_1 == 2.] = 0.
    color_mask_type_1 = color_mask_type_1.fillna(value = 0)

    color_mask_type_2 = color_history_read.copy();
    color_mask_type_2[color_mask_type_2 == 1.] = 0.
    color_mask_type_2[color_mask_type_2 == 2.] = 1.
    color_mask_type_2 = color_mask_type_2.fillna(value = 0) 
    #print(color_mask_type_1)
    #print(color_history_read)

    tp1 = history_read * color_mask_type_1.values
    tp2 = history_read * color_mask_type_2.values

    tp1_array = np.array(tp1) 
    #print(tp1)
    df = pd.DataFrame() 
    for i in range(len(tp1_array)):
        (unique, counts) = np.unique(tp1_array[i], return_counts = True)
        #print(unique)
        unique_str = [str(unique[i]) for i in range(len(unique))]
        #print(unique_str)
        dictionary = {unique_str[i] : [counts[i]] for i in range(len(counts))}
    # print(dictionary)
        data = pd.DataFrame.from_dict(dictionary) 
        
        #print(data)
        df=pd.concat([df,data],axis = 0, ignore_index = True)

    #df = df.fillna(value = 0.)

    tp2_array = np.array(tp2) 
    #print(tp2)
    df2 = pd.DataFrame() 
    for i in range(len(tp2_array)):
        (unique, counts) = np.unique(tp2_array[i], return_counts = True)
        #print(unique)
        unique_str = [str(unique[i]) for i in range(len(unique))]
        #print(unique_str)
        dictionary = {unique_str[i] : [counts[i]] for i in range(len(counts))}
    # print(dictionary)
        data = pd.DataFrame.from_dict(dictionary) 
        
        #print(data)
        df2=pd.concat([df2,data],axis = 0, ignore_index = True)

    #df2 = df2.fillna(value = 0.)
    #print(df2)
    #plt.figure()
    for key in df2.keys(): 
        x = df2.index
        y = float(key)*np.ones(len(x))
        width = df2[key].values
        #plt.plot(x,y*2,color='c')
        plt.fill_between(x,y*3-width,y*3+width,color='c',alpha = 0.75)

    for key in df.keys(): 
        x = df.index
        y = float(key)*np.ones(len(x))
        width = df[key].values
        #plt.plot(x,y*2,color='c')
        plt.fill_between(x,y*3-width,y*3+width,color='r',alpha = 0.5)
        
    plt.ylim([3*7500,3*7800])
