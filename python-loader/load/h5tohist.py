""" Adding data sets of histories to the existing h5 file """ 

import h5py
import pandas as pd 
from auxiliary.analyze_history import *
import os

def write_to_existing_collection(collection_name, filename, h5File = 'pCell_ER_data.h5'):
    """ 
    Writing new data from the simulations into the existing collection of the same name
    
    Input
    -----
    
    collection_name - name of the collection
    
    filename - directory name of what is going to be written
    
    Output
    ------
    
    new data gets written to the collection
    
    """
    
    # check how many datasets collection already has
    if not os.path.exists(h5File):
        
        f = h5py.File(h5File, 'w')
        f.create_group('data')
        
        
    f = h5py.File(h5File, 'r')
    if collection_name in f['data'].keys():
        print('Collection already exists')
        
        # check if there is already a dataset with the same name 
        for dataset in f['data/'+collection_name]:
            #print(dataset)
            if f['data/'+collection_name+'/'+dataset].attrs['filename'] == filename:
                print('Dataset with the same name already exists in this collection') 
                return
        
    else:
        print('Collection did not exist')
        return 
            
    dsets_number = len(f['data/'+collection_name].keys())    
    f.close()
    
    # get the history dataframe 
    history_read = pd.read_hdf(filename,'data/history');
    history_read = history_read.fillna(value=0.)
    R = pd.read_hdf(filename,'data/radius')
    # get growth layer
    growth_layer_read = pd.read_hdf(filename,'data/growth_layer')
    growth_layer_read['rad']=R.values
    growth_layer_read=growth_layer_read.set_index('rad')
    #color mask to distinguish types of sectors
    color_history_read = pd.read_hdf(filename,'data/color_history')
    # get a mask
    masked = mask(history_read, color_history_read)
    # measure width in cell number 
    widths = measure_width(masked)
    # history of blue 
    if len(widths) == 1:
        # history of red 
        df = widths[0]
        df['rad'] = R.values
        df = df.set_index('rad')
        
        # write the file to the collection as new dataset 
        df.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/red')
        growth_layer_read.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/growth_layer')
    elif len(widths) > 1 : 
        df2 = widths[1]
        df2['rad'] = R.values 
        df2 = df2.set_index('rad')
        # history of red 
        df = widths[0]
        df['rad'] = R.values
        df = df.set_index('rad')
        
        # write the file to the collection as new dataset 
        df.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/red')
        df2.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/blue')
        growth_layer_read.to_hdf(h5File, 'data/' + collection_name + '/ds' + str(dsets_number + 1) + '/growth_layer')
    with h5py.File(h5File, 'r+') as ff: 
        ff['data/'+collection_name+'/ds'+str(dsets_number+1)].attrs['filename'] = filename 
    
    return 0

    
def write_to_new_collection(collection_name, filename, h5File = 'pCell_ER_data.h5'):
    """ 
    Writing new data from the simulations into the existing collection of the same name
    
    Input
    -----
    
    collection_name - name of the collection
    
    filename - directory name of what is going to be written
    
    Output
    ------
    
    new data gets written to the collection
    
    """
    
    # check if collection already exists 
    if not os.path.exists(h5File):
        
        f = h5py.File(h5File, 'w')
        f.create_group('data/')
        
    f = h5py.File(h5File, 'r')
    if collection_name in f['data'].keys():
        print('Collection already exists')
        return
    else:
        print('Collection did not exist')
         
            
    dsets_number = 0
    f.close()
    
    # get the history dataframe 
    history_read = pd.read_hdf(filename,'data/history');
    history_read = history_read.fillna(value=0.)
    R = pd.read_hdf(filename,'data/radius')
    # get growth layer
    growth_layer_read = pd.read_hdf(filename, 'data/growth_layer')
    growth_layer_read['rad'] = R.values
    growth_layer_read=growth_layer_read.set_index('rad')
    #color mask to distinguish types of sectors
    color_history_read = pd.read_hdf(filename,'data/color_history');
    # get a mask
    masked = mask(history_read, color_history_read)
    # measure width in cell number 
    widths = measure_width(masked)
    # history of blue 
    if len(widths) == 1:
        # history of red 
        df = widths[0]
        df['rad'] = R.values
        df = df.set_index('rad')
        
        # write the file to the collection as new dataset 
        df.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/red')
        growth_layer_read.to_hdf(h5File, 'data/' + collection_name + '/ds' + str(dsets_number + 1) + '/growth_layer')
    elif len(widths) > 1 : 
        df2 = widths[1]
        df2['rad'] = R.values 
        df2 = df2.set_index('rad')
        # history of red 
        df = widths[0]
        df['rad'] = R.values
        df = df.set_index('rad')
        
        # write the file to the collection as new dataset 
        df.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/red')
        df2.to_hdf(h5File,'data/'+collection_name+'/ds'+str(dsets_number+1)+'/blue')
        growth_layer_read.to_hdf(h5File, 'data/' + collection_name + '/ds' + str(dsets_number + 1) + '/growth_layer')
    with h5py.File(h5File, 'r+') as ff: 
        ff['data/'+collection_name+'/ds'+str(dsets_number+1)].attrs['filename'] = filename 
        
        
    return 0   

