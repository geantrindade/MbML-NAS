import random
import os 
from vislogger import VisLog
from core.load_data import get_runs
import csv
import time
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Nb101Dataset, DatasetPred
from model import NeuralPredictor, NeuralClassifier

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as r2

def get_runs_models():
    runs_n = [4, 12, 36, 108]
    all_runs = {}
    for n_run in runs_n:
        _, runs_p = get_runs(str.format('runs_p/runs%s' % n_run))
        _, runs_c = get_runs(str.format('runs_c/runs%s' % n_run))
        all_runs[str(n_run)] = {'runs_p': runs_p, 'runs_c':runs_c}

    return all_runs


def run_classifier_filter(model_path, sample_train_index, epoch_set, train_split, seed):

    def decision(value):
        if value >= .50:
            return True
        else:
            return False
    
    def convert_prediction(predictions):
        preds = [decision(x) for x in predictions]
        return preds

    gcn_hidden_dict = {'43':48, 
                        '86':72, 
                        '129':96, 
                        '172':144,
                        '344':210,
                        '860': 320}


    epoch_set_dict = {'4':0, 
                        '12':1, 
                        '36':2, 
                        '108':3}

    train_batch_size = 10

    dataset_classifier = Nb101Dataset(split='all', 
                                exclude_split=sample_train_index, 
                                epoch_set=epoch_set_dict[epoch_set], 
                                seed=seed)
    data_loader = DataLoader(dataset_classifier, batch_size=train_batch_size, shuffle=True, drop_last=True)
    
    net_classifier = NeuralClassifier(gcn_hidden=gcn_hidden_dict[train_split])
    net_classifier.load_state_dict(torch.load(model_path))
    net_classifier.eval()
    index_set = []
    total_elem = 0
    for step, batch in enumerate(tqdm(data_loader, desc="Classifier...", \
                 total=len(range(data_loader.__len__())), ascii=False, ncols=100)):
        
        indexes = batch['index'].numpy()
        preds = net_classifier(batch)
        preds = convert_prediction(preds)
        
        index_filter = indexes[preds]
        index_set.append(index_filter) 
        

    index_set = np.asarray(index_set)
    index_set = np.concatenate(index_set)
    return index_set

def run_predictor(sample_from_classifier, model_path, sample_train_index, epoch_set, train_split, seed):
    
    gcn_hidden_dict = {'43':48, 
                        '86':72, 
                        '129':96, 
                        '172':144,
                        '344':210,
                        '860': 320}
    
    epoch_set_dict = {'4':0, 
                        '12':1, 
                        '36':2, 
                        '108':3}

    train_batch_size = 10

    dataset = DatasetPred(split=sample_from_classifier, 
                            exclude_split=sample_train_index, 
                            epoch_set=epoch_set_dict[epoch_set], 
                            seed=seed)
    print(dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    net_pred = NeuralPredictor(gcn_hidden=gcn_hidden_dict[train_split])
    print(net_pred)
    net_pred.load_state_dict(torch.load(model_path))
    net_pred.eval()
    index_set = []
    val_acc_set = []
    test_acc_set = []
    preds_set = []
    total_elem = 0
    for step, batch in enumerate(tqdm(data_loader, desc="Predictor...", \
                 total=len(range(data_loader.__len__())), ascii=False, ncols=100)):

        indexes = batch['index'].numpy()
        real_val_acc = batch['val_acc'].numpy()
        real_test_acc = batch['test_acc'].numpy()
        preds = net_pred(batch)
        
        index_set.append(indexes)
        val_acc_set.append(real_val_acc)
        test_acc_set.append(real_test_acc)
        preds_set.append(preds.detach().numpy())

    index_set = np.asarray(index_set)
    val_acc_set = np.asarray(val_acc_set)
    test_acc_set = np.asarray(test_acc_set)
    preds_set = np.asarray(preds_set)

    index_set = np.concatenate(index_set)
    val_acc_set = np.concatenate(val_acc_set)
    test_acc_set = np.concatenate(test_acc_set)
    preds_set = np.concatenate(preds_set)
    
    df_dict = {'index':index_set, 'preds':preds_set, 'val_acc':val_acc_set, 'test_acc':test_acc_set}
    df = pd.DataFrame(df_dict)
    df = df.sort_values(by=['preds'], ascending=False)
    
    return df



if __name__ == "__main__":

    list_exclude = [] #Insert runs IDs to skip them in the evaluation

    all_models = get_runs_models()
    runs_n = ['4', '12', '36', '108'] 
    train_indexes = str('data/train.npz')
    train_indexes = np.load(train_indexes)

    for epoch_set, r_n in enumerate(runs_n):
        base_dir = str.format('GCN_complete/GCN_complete_runs%s' % r_n)
        try:
            os.mkdir(base_dir)
        except FileExistsError:
            pass
        
        index_path = str('data/classifierIndex_%s.npz' % r_n)
        classifier_indexes = np.load(index_path)
        runs_pred = all_models[r_n]['runs_p']
        runs_clas = all_models[r_n]['runs_c']
        
        for r_p in runs_pred.iterrows():
            
            #coletando os modelos treinados dos preditores
            pred_id_run = r_p[1]['id_run'] 
            if pred_id_run not in list_exclude:
                pred_model_path = r_p[1]['model']
                #coletando o modelo equivalente do classificador
                csf_model_path = runs_clas[runs_clas['id_run'] == pred_id_run]['model'].values[0]
                args = pd.read_csv(runs_clas[runs_clas['id_run'] == pred_id_run]['args'].values[0], header=None)
                
                #Obtendo indexes usados no treino para excluir do conjunto dataset
                train_split = args[1][0]
                seed = args[1][3]
                train_idx_key = str.format('%s_%s' % (seed, train_split))
                train_idx_csf_key = str.format('%s_%s_%s' % (pred_id_run, seed, train_split))
                
                sample_train_idx = train_indexes[train_idx_key]
                sample_train_csf_idx = classifier_indexes[train_idx_csf_key]
                print('\n=======================================\n')
                print('Running setup: [%s] \nEpochs: %s \nSeed: %s \nTrain-Sample: %s \n' \
                            % (pred_id_run, r_n, seed, train_split))
                #para corrigir a execucao em separado do 4
                
                sample_from_classifier = run_classifier_filter(csf_model_path, sample_train_idx, r_n, train_split, seed)
                if np.any(sample_from_classifier):
                    preds = run_predictor(sample_from_classifier, pred_model_path, sample_train_csf_idx, r_n, train_split, seed)
                    #preds = run_predictor('all', pred_model_path, sample_train_idx, epoch_set, train_split, seed)
                    
                    print('\n=======================================\n')
                    
                    csv_res_path = str.format('%s/preds%s_%s_%s.csv' % (base_dir, pred_id_run, seed, train_split))
                    preds.to_csv(csv_res_path)
                    
