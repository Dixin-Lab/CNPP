import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import constant
import utils

from tqdm import tqdm


def train_epoch(model, training_data_list, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    
    # training_data_list: [dataloader0, dataloader1]
    # if process_idx == 1:     event_type -= model.num_types[0] !!!
    training_data_list = [iter(dataloader) for dataloader in training_data_list]

    for process_idx, dataloader in enumerate(training_data_list):
    # for batch in tqdm(training_data, mininterval=2,
    #                   desc='  - (Training)   ', leave=False):
        batch = next(dataloader)

        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(process_idx, event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = utils.log_likelihood(model, process_idx, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(constant.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(constant.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

# TODO
def eval_epoch(model, validation_data_list, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():

        # training_data_list: [dataloader0, dataloader1]
        # if process_idx == 1:     event_type -= model.num_types[0] !!!
        validation_data_list = [iter(dataloader) for dataloader in validation_data_list]

        for process_idx, dataloader in enumerate(validation_data_list):
        # for batch in tqdm(validation_data, mininterval=2,
        #                   desc='  - (Validation) ', leave=False):
            batch = next(dataloader)

            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(constant.PAD).sum().item()
            total_num_pred += event_type.ne(constant.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train(model, training_data_list, validation_data_list, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data_list, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
            'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = eval_epoch(model, validation_data_list, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
            'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
            'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
            .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()