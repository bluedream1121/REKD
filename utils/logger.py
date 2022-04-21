r"""Logging"""
import datetime
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.exp_name +  logtime + '_group' + str(args.group_size) + '_f' + str(args.dim_first) + '_s' + str(args.dim_second) + '_t' + str(args.dim_third)
        cls.logpath = os.path.join(args.log_dir  +'/' +  logpath  + '.log')

        os.makedirs(args.log_dir, exist_ok=True) 
        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # # Log arguments
        logging.info('\n+=========== Rotation-Equivariant Keypoint Detection  ============+')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('+================================================+\n')
        logging.info('Save log path : %s ' % (cls.logpath))

        return logtime, logpath

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_model(cls, model, epoch, val_rep):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model_'+str(epoch)+'.pt'))
        cls.info('Model saved @%d w/ val. Repeability score: %5.2f.\n' % (epoch, val_rep))

    @classmethod
    def save_best_model(cls, model, val_rep):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model'+'.pt'))
        cls.info('Best Model saved w/ val. Repeability score: %5.2f.\n' % (val_rep))


class AverageMeter:
    r""" Stores loss, evaluation results,"""

    def __init__(self, training=False):
        if training:
            self.buffer_keys = ['key_loss', 'ori_loss', 'total_loss']
        else:
            self.buffer_keys = ['repeatability', 'ori_acc', 'ori_apx_acc']
        self.buffer = {}
        
        for key in self.buffer_keys:
            self.buffer[key] = []

        self.split = 'Train' if training else 'Valid'
    
    def update(self, eval_result):
        for key in self.buffer_keys:
            self.buffer[key].append(eval_result[key])

    
    def write_result(self, epoch):
        msg = '*** [@Epoch %02d] ' % epoch

        for key in self.buffer_keys:
            msg += '%s: %6.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))

        msg += '***\n'
        Logger.info(msg)
    
    def write_process(self, batch_idx, datalen, epoch):
        msg = '[Epoch: %02d] ' % epoch
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        msg += '[%s] ' % (self.split)        

        for key in self.buffer_keys:
            msg += 'Avg %s: %5.2f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]) )
        # Logger.info(msg)
        return msg

    def get_results(self):
        result = {}
        for key in self.buffer_keys:
            result[key] = torch.stack(self.buffer[key]).mean(dim=0)

        return result
        
    def get_test_results(self):
        result = {}
        for key in self.buffer_keys:
            result[key] = torch.stack(self.buffer[key]).mean(dim=0)

        return result


class Recorder:
    def __init__(self):
        self.buffer_keys = ['repeatability', 'ori_acc', 'ori_apx_acc', 'total_loss']
        self.buffer = {}
        
        for key in self.buffer_keys:
            self.buffer[key] = []

        self.best_epoch = -1
    
    def update(self, epoch, eval_result):
        for key in self.buffer_keys:
            self.buffer[key].append(eval_result[key])

        Logger.info((' Epoch {} (val) : rep_s: {:.2f}, ori: {:.2f}/{:.2f}, loss: {:.2f}. '.format(epoch, \
            eval_result['repeatability'], eval_result['ori_acc'], eval_result['ori_apx_acc'], eval_result['total_loss'])))

        ## best model policy is repeatability score.
        if eval_result['repeatability'] >= max(self.buffer['repeatability']):
            self.best_epoch = epoch
            
            return True
        return False

    def get_results(self):
        result = {}
        for key in self.buffer_keys:
            result[key] = self.buffer[key][self.best_epoch]

        return self.best_epoch, result

