# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import pandas as pd
import numpy as np
import argparse
import os
import tensorflow.compat.v1 as tf

import model
import evaluation

class Args():
    is_training = True
    layers = 1
    rnn_size = 100
    n_epochs = 5
    batch_size = 256
    dropout_p_hidden= 0.3
    learning_rate = 0.05
    initial_accumulator_value = 1e-1
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--m', '--measure', type=int, nargs='+', default=[20])

    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument('--cuda_dev_id', type=str, default='0')    
    parser.add_argument('--decay', default=1.0, type=float)
    parser.add_argument('--decay_steps', default=1e4, type=float)
    parser.add_argument('--initial_accumulator_value', default=1e-1, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(command_line.train_path, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(command_line.test_path, sep='\t', dtype={'ItemId': np.int64})
    args = Args()
    args.n_items = len(data['ItemId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.batch_size = command_line.batch_size
    args.decay = command_line.decay
    args.decay_steps = command_line.decay_steps
    args.checkpoint_dir = command_line.checkpoint_dir
    args.initial_accumulator_value = command_line.initial_accumulator_value
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else 1 - command_line.dropout # command_line.dropout # 1 - command_line.dropout !!!
    print(pd.DataFrame({'Args':list(args.__dict__.keys()), 'Values':list(args.__dict__.values())}))
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data)
        else:
            for c in command_line.m:
                res = evaluation.evaluate_sessions_batch(gru, data, valid, cut_off=c, batch_size=command_line.batch_size)
                print('Recall@{}: {:.8} MRR@{}: {:.8}'.format(c, res[0], c, res[1]))