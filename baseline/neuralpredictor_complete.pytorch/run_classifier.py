import logging
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

from dataset import Nb101Dataset
from model import NeuralPredictor, NeuralClassifier
from utils import AverageMeter, AverageMeterGroup, get_logger, reset_seed, to_cuda

from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as r2


def run_classifier(args):

    def decision(value):
            if value >= .50:
                return True
            else:
                return False
        
    def convert_prediction(predictions):
        preds = [decision(x) for x in predictions]
        return preds

    runs_n = ['4', '12', '36','108']

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

    for epoch_set, r_n in enumerate(runs_n):
        runs_folder = 'runs_c/runs' + r_n
        args.trainset_epochs = epoch_set_dict[r_n]
        
        df_meta, df_runs = get_runs(runs_folder)
       
        seeds = [0, 1, 10, 42, 100, 123, 666, 1000, 1234, 12345]
        train_sizes = ['43', '86', '129', '172', '344', '860']
        id_run = 0
        dict_array = {}
        for seed in seeds:
            for train_size in train_sizes:
                args.gcn_hidden = gcn_hidden_dict[train_size]
                train_size_seed = str.format('%d_%s' % (seed, train_size))
                id_r = str('%05d' % id_run)
                print('seed', seed)
                print('train_size', train_size)
                print('id_r', id_r)
                dataset_classifier = Nb101Dataset(split='all', exclude_split=train_size_seed, epoch_set=epoch_set_dict[r_n], seed=seed)
                
                data_loader = DataLoader(dataset_classifier, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
                net_classifier = NeuralClassifier(gcn_hidden=args.gcn_hidden)
                model_path = df_runs[df_runs.id_run == id_r]['model'].values[0]
                net_classifier.load_state_dict(torch.load(model_path))
                net_classifier.eval()
                index_set = []
                total_elem = 0
                for step, batch in enumerate(data_loader):
                    indexes = batch['index'].numpy()
                    preds = net_classifier(batch)
                    preds = convert_prediction(preds)
                    index_filter = indexes[preds]
                    total_elem += len(index_filter)
                    index_set.append(index_filter)
                    
                    if total_elem >= int(train_size):
                        index_set = np.asarray(index_set)
                        index_set = np.concatenate(index_set)
                        index_set = index_set[:int(train_size)]
                        break
                
                label = str.format('%s_%d_%s' % (id_r, seed, train_size)) 
                dict_array[label] = index_set
                id_run += 1
                   
        path = str.format('data/classifierIndex_%s.npz' % (r_n))
        np.savez(path, **dict_array)
        


if __name__ == "__main__":

    valid_splits = ["43", "86", "129", "172", "334", "860", "91-172", "91-334", "91-860", "denoise-91", "denoise-80", "all"]
    parser = ArgumentParser()
    parser.add_argument("--train_split", choices=valid_splits, default="172")
    parser.add_argument("--eval_split", choices=valid_splits, default="all")
    parser.add_argument("--gcn_hidden", type=int, default=144)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train_batch_size", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=1000, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=1e-3, type=float)
    parser.add_argument("--train_print_freq", default=None, type=int)
    parser.add_argument("--eval_print_freq", default=10, type=int)
    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--trainset_epochs", default=4, type=int)
    args = parser.parse_args()
    
    run_classifier(args)

