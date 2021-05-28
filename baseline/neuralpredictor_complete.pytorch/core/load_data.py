import numpy as np
import pandas as pd
import sys
import os
import csv
import pathlib



def get_files_path(id_run, files_path):
    
    d_path = {}
    train_path, val_path, test_path, args_path, model_path = None, None, None, None, None
    for r, s, files in os.walk(files_path):
        for file in files:
            if 'train' in file:
                train_path = str(files_path) +'/'+ file
            elif 'val' in file:
                val_path = str(files_path) +'/'+ file
            elif 'test' in file:
                test_path = str(files_path) +'/'+ file
            elif 'args' in file:
                args_path = str(files_path) +'/'+ file
            elif 'model' in file:
                model_path = str(files_path) +'/model/'+ file
            d_path = {'id_run':id_run, 'args':args_path, 'train':train_path, 'val':val_path, 'test': test_path, 'model':model_path}
    
    return d_path

def get_runs(runs_folder):
    abs_path = pathlib.Path.joinpath(pathlib.Path(__file__).parent.parent.absolute(), runs_folder)
    df_runs = pd.DataFrame(columns=['id_run', 'args', 'train', 'val', 'test', 'model'])
    for root, subfolders, files in os.walk(abs_path):
        spl = []
        for i, folder in enumerate(subfolders):
            files_path = pathlib.Path.joinpath(abs_path, folder)
            folder = 'View_' + folder
            sp = str.split(folder, '_')
            spl.append(sp)
            
            
            d_path_run = get_files_path(sp[1], files_path)
            df_runs = df_runs.append(d_path_run, ignore_index=True)
        spl = np.asarray(spl)
        df_meta_runs = {'act':spl[:,0], 'id':spl[:, 1], 'datetime':spl[:, 2]}
        return pd.DataFrame(df_meta_runs), df_runs


#def get_data_graph_from_run():
