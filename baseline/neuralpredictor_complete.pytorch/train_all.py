import logging
import random
import os 
from vislogger import VisLog
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
from model import NeuralPredictor
from utils import AverageMeter, AverageMeterGroup, get_logger, reset_seed, to_cuda

from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as r2

def accuracy_mse(predict, target, scale=100.):
    predict = predict.detach()# * scale
    target = target# * scale
    return F.mse_loss(predict, target)


def visualize_scatterplot(predict, target, scale=100.):
    def _scatter(x, y, subplot, threshold=None):
        plt.subplot(subplot)
        plt.grid(linestyle="--")
        plt.xlabel("Validation Accuracy")
        plt.ylabel("Prediction")
        plt.scatter(target, predict, s=1)
        if threshold:
            ax = plt.gca()
            ax.set_xlim(threshold, 95)
            ax.set_ylim(threshold, 95)
    predict = Nb101Dataset.denormalize(predict) * scale
    target = Nb101Dataset.denormalize(target) * scale
    plt.figure(figsize=(12, 6))
    _scatter(predict, target, 121)
    _scatter(predict, target, 122, threshold=90)
    plt.savefig("assets/scatterplot.png", bbox_inches="tight")
    plt.close()


def main(base_dir, run_dir_name, n_e, id_run, seed, train_size, classifier_filter, epoch_set):

    path_logs = str.format('%s/%s/' % (base_dir, run_dir_name))

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

    vis_log = VisLog(run_dir_name, path_logs, 100, 'tt')

    args = parser.parse_args()

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

    args.epochs = n_e
    args.train_split = train_size
    args.seed = seed
    args.trainset_epochs = epoch_set
    args.gcn_hidden = gcn_hidden_dict[train_size]

    args_values = vars(args)
    vis_log.log_arguments(args_values)
    
    reset_seed(args.seed)
    train_split_seed = str.format('%d_%s' % (seed, train_size))
    #dataset = Nb101Dataset(split=train_split_seed)
    
    dataset = DatasetPred(split=classifier_filter, epoch_set=epoch_set_dict[epoch_set], seed=seed)
    dataset_test = Nb101Dataset(split=args.eval_split, exclude_split=classifier_filter, epoch_set=epoch_set_dict[epoch_set], seed=seed)
    data_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset_test, batch_size=args.eval_batch_size)
    net = NeuralPredictor(gcn_hidden=args.gcn_hidden)
    print(net)
    #net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger = get_logger()
    #logger_csv = []
    
    net.train()

    all_target = []
    all_predict = []
    count_plot = 0
    pbar = tqdm(range(args.epochs), desc="Training...", total=len(range(args.epochs)), ascii=False, ncols=150)
    for epoch in pbar:
        start = time.time()
        meters = AverageMeterGroup()
        lr = optimizer.param_groups[0]["lr"]
        batch_len = 0
        mse_sum = 0
        mae_sum = 0
        loss_sum = 0
        target_run, pred_run = [], []
        for step, batch in enumerate(data_loader):
            #batch = to_cuda(batch)
            target = batch["val_acc"].type(torch.float64)
            batch_len += target.size(0)
            predict = net(batch)
            all_target.append(target)
            all_predict.append(predict.detach().numpy())

            target_run.append(target)
            pred_run.append(predict.detach().numpy())

            optimizer.zero_grad()
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict, target)
            loss_sum += loss.detach().item() * target.size(0)
            mse_sum += mse.item() * target.size(0)
            mae_sum += MAE(target, predict.detach().numpy()) * target.size(0)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            '''
            if (args.train_print_freq and step % args.train_print_freq == 0) or \
                    step + 1 == len(data_loader):
                logger.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
                            epoch + 1, args.epochs, step + 1, len(data_loader), lr, meters)
            '''
            pbar.set_postfix({'meters': meters})

        lr_scheduler.step()
        target_run = np.concatenate(target_run)
        pred_run = np.concatenate(pred_run)
        r2_m = r2(target_run, pred_run)
        vis_log.update('train', {'epoch':epoch, 
                                'loss':loss_sum/batch_len, 
                                'mse':mse_sum/batch_len, 
                                'mae':mae_sum/batch_len, 
                                'r2':r2_m,
                                'time':time.time() - start})
        
    torch.save(net.state_dict(), path_logs+'/model/model.pth')

    net.eval()
    meters = AverageMeterGroup()
    predict_, target_ = [], []
    logger_csv = []
    with torch.no_grad():
        batch_len = 0
        mse_sum = 0
        mae_sum = 0
        loss_sum = 0
        pbar_t = tqdm(enumerate(test_data_loader), desc="Testing... ", total=len(test_data_loader), ascii=False, ncols=150)
        for step, batch in pbar_t:
            start = time.time()
            #batch = to_cuda(batch)
            target = batch["val_acc"]
            predict = net(batch)
            predict_.append(predict.cpu().numpy())
            target_.append(target.cpu().numpy())
            mse = accuracy_mse(predict, target)
            batch_len += target.size(0)
            loss_sum += criterion(predict, target).item() * target.size(0)
            mse_sum += mse.item() * target.size(0)
            mae_sum += MAE(target, predict.detach().numpy()) * target.size(0)
            logger_csv.append([step,' ', mse_sum/batch_len, mae_sum/batch_len])
            meters.update({"loss": criterion(predict, target).item(),
                           "mse": accuracy_mse(predict, target).item()}, n=target.size(0))
            '''
            if (args.eval_print_freq and step % args.eval_print_freq == 0) or \
                    step % 10 == 0 or step + 1 == len(test_data_loader):
                logger.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_data_loader), meters)
            '''
            pbar_t.set_postfix({'meters':meters})
            r2_m = r2(target, predict)
            
            vis_log.update('test', {'epoch':step, 
                                'loss':loss_sum/batch_len,
                                'mse':mse_sum/batch_len, 
                                'mae':mae_sum/batch_len, 
                                'r2':r2_m,
                                'time': time.time() - start})

    predict_ = np.concatenate(predict_)
    target_ = np.concatenate(target_)
    ken = kendalltau(predict_, target_)
    logger.info("Kendalltau: %.6f || p = %.6f" % (ken[0], ken[1]) )
    if args.visualize:
        visualize_scatterplot(predict_, target_)


if __name__ == "__main__":

    runs_n = ['4', '12', '36','108']

    for epoch_set, r_n in enumerate(runs_n):
        base_dir = 'runs_p/runs' + r_n
        
        try:
            os.mkdir(base_dir)
        except FileExistsError:
            pass
        epochs = [300]
        seeds = [0, 1, 10, 42, 100, 123, 666, 1000, 1234, 12345]
        train_sizes = ['43', '86', '129', '172', '344', '860']
        id_run = 0
        total_runs = len(epochs) * len(seeds) * len(train_sizes)
        for n_e in epochs:
            for seed in seeds:
                for train_size in train_sizes:
                    split_name = str.format('%05d_%d_%s' % (id_run, seed, train_size))
                    classifier_filter_file = str.format('data/classifierIndex_%s.npz' % r_n)
                    classifier_filter = np.load(classifier_filter_file)[split_name]
                    
                    if np.any(classifier_filter):
                        run_dir_name = str.format('%05d_%s' % (id_run, datetime.now()))
                        try:
                            os_path = os.path.join(base_dir, run_dir_name)
                            os.mkdir(os_path)
                            os_path = os.path.join(os_path, 'model')
                            os.mkdir(os_path)
                        except FileExistsError:
                            pass
                        print('\n=======================================\n')
                        print('Running setup: [%d/%d] \nEpochs: %d \nSeed: %d \nTrain-Sample: %s \n' \
                                % (id_run+1, total_runs, n_e, seed, train_size))
                        main(base_dir, run_dir_name, n_e, id_run, seed, train_size, classifier_filter, r_n)
                        print('\n=======================================\n')
                        
                    id_run += 1
        
       