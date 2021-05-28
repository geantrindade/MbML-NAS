import numpy as np
import pandas as pd
import csv

class VisLog:

    def __init__(self, id, filepath, save_freq=1, log_level='tvt'):
        self.id = id
        self.path = filepath
        w, csv = self.create_persistent_files(filepath, log_level)
        self.w_train, self.w_val, self.w_test = w
        train, val, test = csv
        self.act_flag = {'train':False, 'val':False, 'test':False}
        self.w = {'train':self.w_train, 'val': self.w_val, 'test':self.w_test}
        self.csv = {'train':train, 'val':val, 'test':test}
        self.logger = []
        self.counters = {'train':1, 'val':1, 'test':1}
        self.save_freq = save_freq

    def update(self, logger, values_dict):

        if self.act_flag[logger]:
            self.logger.append(list(values_dict.values()))
            if not (self.counters[logger] % self.save_freq == 0):
                self.w[logger].writerows(self.logger)
                self.logger.clear()
                self.csv[logger].flush()
                self.counters[logger] = 1
            self.counters[logger] += 1 #len(values_dict)
        else:
            self.logger.clear()
            self.w[logger].writerow(list(values_dict.keys()))
            self.act_flag[logger] = True

    def log_arguments(self, args):
        with open(str.format('%s/argsparse.csv' % (self.path)), 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in args.items():
                writer.writerow([key, value])


    def create_persistent_files(self, filepath, log_level):
        w_train = None
        w_val = None
        w_test = None

        train, val, test = None, None, None

        if log_level == 'tvt':
            train = open(str.format('%s/train_vislog.csv' % (filepath)), 'a')
            val = open(str.format('%s/val_vislog.csv' % (filepath)), 'a')
            test = open(str.format('%s/test_vislog.csv' % (filepath)), 'a')
            w_train = csv.writer(train, delimiter=',')
            w_val = csv.writer(val, delimiter=',')
            w_test = csv.writer(test, delimiter=',')
        elif log_level == 't':
            train = open(str.format('%s/train_vislog.csv' % (filepath)), 'a')
            w_train = csv.writer(train, delimiter=',')
        elif log_level == 'tv':
            train = open(str.format('%s/train_vislog.csv' % (filepath)), 'a')
            val = open(str.format('%s/val_vislog.csv' % (filepath)), 'a')    
            w_train = csv.writer(train, delimiter=',')
            w_val = csv.writer(val, delimiter=',')
        elif log_level == 'tt': 
            train = open(str.format('%s/train_vislog.csv' % (filepath)), 'a')
            test = open(str.format('%s/test_vislog.csv' % (filepath)), 'a')
            w_train = csv.writer(train, delimiter=',')
            w_test = csv.writer(test, delimiter=',')

        return (w_train, w_val, w_test), (train, val, test)

    