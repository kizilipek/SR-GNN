#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from src.model import Model
from src.utils import Data, get_configs
import pickle
import datetime


def main():
    env_file = "dev"
    print("program started with {} confguration".format(env_file))
    cfg = get_configs(env_file)  # fetch configurations
    model_params = cfg['model']
    #load the data
    train_data = pickle.load(open('../datasets/' + model_params['dataset'] + '/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + model_params['dataset'] + '/test.txt', 'rb'))
    if model_params['dataset'] == 'diginetica':
        n_node = 43098
    elif model_params['dataset'] == 'yoochoose1_64' or model_params['dataset'] == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    train_data = Data(train_data, sub_graph=True, shuffle=True)
    test_data = Data(test_data, sub_graph=True, shuffle=False)
    model = Model(hidden_size=model_params['hiddenSize'], out_size=model_params['hiddenSize'], batch_size= model_params['batchSize'], n_node=n_node,
                  lr=model_params['lr'], l2=model_params['l2'], step=model_params['step'])
    best_result = [0, 0]
    best_epoch = [0, 0]
    for epoch in range(model_params['epoch']):
        print('epoch: ', epoch, '===========================================')
        slices = train_data.generate_batch(model.batch_size)
        fetches = [model.opt, model.loss_train, model.global_step]
        print('start training: ', datetime.datetime.now())
        loss_ = []
        for i, j in zip(slices, np.arange(len(slices))):
            adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)

            _, loss, _ = model.sess.run(fetches, feed_dict={model.tar: targets, model.item: item, model.adj_in: adj_in,
                                                            model.adj_out: adj_out, model.alias: alias,
                                                            model.mask: mask})

            loss_.append(loss)
        loss = np.mean(loss_)
        slices = test_data.generate_batch(model.batch_size)
        print('start predicting: ', datetime.datetime.now())
        hit, mrr, test_loss_ = [], [], []
        for i, j in zip(slices, np.arange(len(slices))):
            adj_in, adj_out, alias, item, mask, targets = test_data.get_slice(i)

            scores, test_loss = model.sess.run([model.score_test, model.loss_test],
                                               feed_dict={model.tar: targets, model.item: item, model.adj_in: adj_in,
                                                          model.adj_out: adj_out, model.alias: alias, model.mask: mask})
            test_loss_.append(test_loss)
            index = np.argsort(scores, 1)[:, -20:]
            for score, target in zip(index, targets):
                hit.append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (20 - np.where(score == target - 1)[0][0]))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        test_loss = np.mean(test_loss_)
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
        print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' %
              (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))


if __name__ == '__main__':
    main()

