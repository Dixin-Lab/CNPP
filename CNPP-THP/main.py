import random
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from data_util import get_dataloader
from constant import PAD
from utils import LabelSmoothingLoss
from utils import log_likelihood
from model import Transformer


def prepare_dataloader(opt, train_ratio=0.8):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
            num_types = [data['pmat'].shape[0], data['pmat'].shape[1]]

            return data, num_types

    # print('[Info] Loading train data...')
    data, num_types = load_data(opt.data)
    params1 = data['params1']
    params2 = data['params2']
    pmat = data['pmat']

    total_data0 = data['seq1'][:opt.seq_num]
    total_data1 = data['seq2'][:opt.seq_num]

    len0 = len(total_data0)
    len1 = len(total_data1)

    train_len0 = int(len0 * train_ratio)
    test_len0 = int(len0 * (1 - train_ratio) / 2)

    train_len1 = int(len1 * train_ratio)
    test_len1 = int(len1 * (1 - train_ratio) / 2)

    trainloader_0 = get_dataloader(
        data=total_data0[:train_len0], batch_size=opt.batch_size)
    trainloader_1 = get_dataloader(
        data=total_data1[:train_len1], batch_size=opt.batch_size)
    testloader_0 = get_dataloader(
        data=total_data0[train_len0:test_len0 + train_len0], batch_size=opt.batch_size)
    testloader_1 = get_dataloader(
        data=total_data1[train_len1:test_len1 + train_len1], batch_size=opt.batch_size)

    valloader_0 = get_dataloader(
        data=total_data0[test_len0 + train_len0:], batch_size=opt.batch_size)
    valloader_1 = get_dataloader(
        data=total_data1[test_len1 + train_len1:], batch_size=opt.batch_size)

    return valloader_0, valloader_1, trainloader_0, trainloader_1, testloader_0, testloader_1, num_types, params1, params2, pmat


def train_epoch(model, training_data_list, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_num_event = 0  # number of total events

    total_event_ll_0 = 0  # cumulative event log-likelihood
    total_num_event_0 = 0  # number of total events

    total_event_ll_1 = 0  # cumulative event log-likelihood
    total_num_event_1 = 0  # number of total events

    training_data_iters = [iter(dataloader)
                           for dataloader in training_data_list]

    training_data_batch_nums = min(
        [len(dataloader) for dataloader in training_data_list])

    # print(training_data_batch_nums)

    # for batch_num in range(training_data_batch_nums):
    for batch_num in tqdm(range(training_data_batch_nums), mininterval=2, desc='  - (Training)   ', leave=False):

        batch_num_event = 0  # batch_num_event
        batch_event_loss = 0

        """ forward """
        optimizer.zero_grad()

        for process_idx in range(2):

            batch = next(training_data_iters[process_idx])
            """ prepare data """
            event_time, time_gap, event_type = map(
                lambda x: x.to(opt.device), batch)

            enc_out, prediction = model(process_idx, event_type, event_time)

            """ backward """
            # negative log-likelihood
            event_ll, non_event_ll = log_likelihood(
                model, process_idx, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)

            scale_nll_loss = event_type.ne(PAD).sum().item()

            batch_event_loss += event_loss
            batch_num_event += scale_nll_loss  # batch_num_event

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_num_event += event_type.ne(PAD).sum().item()

            """ note keeping process_idx """
            if process_idx == 0:
                total_event_ll_0 += -event_loss.item()
                total_num_event_0 += event_type.ne(PAD).sum().item()
            else:
                total_event_ll_1 += -event_loss.item()
                total_num_event_1 += event_type.ne(PAD).sum().item()

        if opt.isKL:
            trans = model.encoder.event_emb.sinkhorn().T
            trans = trans / torch.sum(trans)

            trans[trans == 0] = 1

            KL_div = nn.KLDivLoss(reduction="sum")

            ot_loss = KL_div(trans.log(), model.encoder.event_emb.P)

            loss = batch_event_loss / batch_num_event + ot_loss * opt.KL_tau

        else:
            loss = batch_event_loss / batch_num_event

        loss.backward()

        """ update parameters """
        optimizer.step()

    ll0 = total_event_ll_0 / total_num_event_0
    ll1 = total_event_ll_1 / total_num_event_1
    ll = (ll0 + ll1) / 2
    print("train epoch process 0", ll0)
    print("train epoch process 1", ll1)
    return ll


def eval_epoch(model, validation_data_list, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_num_event = 0  # number of total events

    total_event_ll_0 = 0  # cumulative event log-likelihood
    total_num_event_0 = 0  # number of total events

    total_event_ll_1 = 0  # cumulative event log-likelihood
    total_num_event_1 = 0  # number of total events

    with torch.no_grad():

        validation_data_iters = [iter(dataloader)
                                 for dataloader in validation_data_list]

        validation_data_batch_nums = min(
            [len(dataloader) for dataloader in validation_data_list])

        print(validation_data_batch_nums)

        # for batch_num in range(training_data_batch_nums):
        for batch_num in tqdm(range(validation_data_batch_nums), mininterval=2, desc='  - (Training)   ', leave=False):

            for process_idx in range(2):

                batch = next(validation_data_iters[process_idx])
                """ prepare data """
                event_time, time_gap, event_type = map(
                    lambda x: x.to(opt.device), batch)

                """ forward """
                enc_out, prediction = model(
                    process_idx, event_type, event_time)

                """ compute loss """
                event_ll, non_event_ll = log_likelihood(
                    model, process_idx, enc_out, event_time, event_type)
                event_loss = -torch.sum(event_ll - non_event_ll)

                """ note keeping """
                total_event_ll += -event_loss.item()
                total_num_event += event_type.ne(PAD).sum().item()

                """ note keeping process_idx """
                if process_idx == 0:
                    total_event_ll_0 += -event_loss.item()
                    total_num_event_0 += event_type.ne(PAD).sum().item()
                else:
                    total_event_ll_1 += -event_loss.item()
                    total_num_event_1 += event_type.ne(PAD).sum().item()

    ll0 = total_event_ll_0 / total_num_event_0
    ll1 = total_event_ll_1 / total_num_event_1
    ll = (ll0 + ll1) / 2

    print("test epoch process 0", ll0)
    print("test epoch process 1", ll1)

    return ll


def train(model, training_data_list, validation_data_list, test_data_list, optimizer, scheduler, pred_loss_func: list,
          opt, gnd_pair):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    test_event_losses = []  # validation log-likelihood

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event = train_epoch(
            model, training_data_list, optimizer, pred_loss_func, opt)
        print('  -Total (Training)    loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event = eval_epoch(
            model, validation_data_list, pred_loss_func, opt)
        print('  -Total (Valid)     loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        print('  - [Val Info] Maximum ll: {event: 8.5f}, '
              .format(event=max(valid_event_losses)))

        test_event = eval_epoch(
            model, test_data_list, pred_loss_func, opt)
        print('  -Total (Testing)     loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=test_event, elapse=(time.time() - start) / 60))

        test_event_losses += [test_event]
        print('  - [Test Info] Maximum ll: {event: 8.5f}, '
              .format(event=max(test_event_losses)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('val {epoch}, {ll: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event))

            f.write('test {epoch}, {ll: 8.5f}\n'
                    .format(epoch=epoch, ll=test_event))

        for topk in [1, 3, 5, 10, 30, 50, 100]:
            score = acc_score_P(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy(), gnd=gnd_pair[:, ],
                                topk=topk)

            with open(opt.log, 'a') as f:
                f.write('top {top}, {score: 8.5f}\n'
                        .format(top=topk, score=score))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./')
    parser.add_argument('-prior', default='./')
    parser.add_argument('-epoch', type=int, default=15)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)

    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_rnn', type=int, default=256)

    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log_path')

    parser.add_argument('-isKL', type=bool, default=True)
    parser.add_argument('-is_param', type=bool, default=True)
    parser.add_argument('-KL_tau', type=float, default=100)
    parser.add_argument('-seq_num', type=int, default=2000)
    opt = parser.parse_known_args()[0]

    # default device is CUDA
    opt.device = torch.device('cuda')

    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    valloader_0, valloader_1, \
    trainloader_0, trainloader_1, \
    testloader_0, testloader_1, \
    num_types, params1, params2, pmat = prepare_dataloader(opt)

    training_data_list = [trainloader_0, trainloader_1]
    validation_data_list = [valloader_0, valloader_1]
    test_data_list = [testloader_0, testloader_1]

    """ read P_prior """
    P_prior = torch.from_numpy(np.load(opt.prior)["P"]).to(opt.device).float()

    gnd_pair = np.argwhere(pmat.numpy()).astype(np.int32)
    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        P_prior=P_prior,
        is_param=opt.is_param
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    # TO DO
    if opt.smooth > 0:
        pred_loss_func0 = LabelSmoothingLoss(
            opt.smooth, num_types[0], ignore_index=-1)
        pred_loss_func1 = LabelSmoothingLoss(
            opt.smooth, num_types[1], ignore_index=-1)
        pred_loss_func = [pred_loss_func0, pred_loss_func1]
    else:
        pred_loss_func0 = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none')
        pred_loss_func1 = nn.CrossEntropyLoss(
            ignore_index=-1, reduction='none')
        pred_loss_func = [pred_loss_func0, pred_loss_func1]

    train(model, training_data_list, validation_data_list, test_data_list, optimizer, scheduler, pred_loss_func, opt,
          gnd_pair)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def topk_alignment_score(sim, gnd, topk, right=1):
    """
    :param sim: n1xn2 similarity matrix (ndarray)
    :param gnd: numx2 the gnd pairs, and the label is beginning with 0.
    :param topk:
    :return: accuracy scores
    """
    possible_alignment = np.argsort(sim, axis=1)
    num = 0
    length = gnd.shape[0]
    for idx in range(length):
        if gnd[idx, right] in possible_alignment[gnd[idx, 1 - right]][-topk:]:
            num += 1
    return num / length


def acc_score_P(P, gnd, topk=5):
    dis = P
    score = topk_alignment_score(dis, gnd, topk, right=1)
    return score


if __name__ == '__main__':
    main()