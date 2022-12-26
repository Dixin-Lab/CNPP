import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

import numpy as np
import random

from models.sahp import SAHP
from utils import atten_optimizer
from utils import util
from utils import reg


def make_model(nLayers=6, d_model=128, atten_heads=8, dropout=0.1, process_dim=10,
               device = 'cpu', pe='concat', max_sequence_length=4096):
    "helper: construct a models form hyper parameters"

    model = SAHP(nLayers, d_model, atten_heads, dropout=dropout, process_dim=process_dim, device = device,
                 max_sequence_length=max_sequence_length)

    # initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self,src,pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt,pad,device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask


def l1_loss(model):
    ## l1 loss
    l1 = 0
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1

def eval_sahp(batch_size, loop_range, seq_lengths_list, seq_times_list, seq_types_list, model, device, lambda_l1=0):
    model.eval()
    epoch_loss = 0
    process_num = len(seq_lengths_list)
    for i in range(process_num):
        batch_epoch_loss = 0.0
        for i_batch in loop_range:
            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
                util.get_batch(batch_size, i_batch, model.process_dim[i], seq_lengths_list[i], seq_times_list[i], \
                                seq_types_list[i], rnn=False)
            batch_seq_types = batch_seq_types[:, 1:]

            masked_seq_types = MaskBatch(batch_seq_types,pad=model.process_dim[i], device=device)# exclude the first added event
            model.forward(i, batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
            nll = model.compute_loss(i, batch_seq_times, batch_onehot)

            batch_epoch_loss += nll.detach()
    event_num = sum([torch.sum(seq_lengths).float() for seq_lengths in seq_lengths_list])
    epoch_loss += batch_epoch_loss
    model.train()
    return event_num, epoch_loss


def train_eval_sahp(params):

    args, process_dim_list, device, tmax_list, max_seq_length_list, \
    train_seq_times_list, train_seq_types_list, train_seq_lengths_list, \
    dev_seq_times_list, dev_seq_types_list, dev_seq_lengths_list, \
    test_seq_times_list, test_seq_types_list, test_seq_lengths_list, \
    batch_size, epoch_num, use_cuda, train_pairs, test_pairs = params

    process_num = len(tmax_list)
    max_sequence_length = max(max_seq_length_list)

    d_model = args.d_model
    atten_heads = args.atten_heads
    dropout = args.dropout

    model = make_model(nLayers=args.nLayers, d_model=d_model, atten_heads=atten_heads,
                    dropout=dropout, process_dim=process_dim_list, device=device, pe=args.pe,
                    max_sequence_length=max_sequence_length + 1).to(device)

    print("the number of trainable parameters: " + str(util.count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.lambda_l2)
    model_opt = atten_optimizer.NoamOpt(args.d_model, 1, 100, initial_lr=args.lr, optimizer=optimizer)

    ## Size of the traing dataset
    # 取list两个元素size的最大值
    train_size = max([train_seq_lengths.size(0) for train_seq_lengths in train_seq_lengths_list])
    dev_size = max([dev_seq_lengths.size(0) for dev_seq_lengths in dev_seq_lengths_list])
    test_size = max([test_seq_lengths.size(0) for test_seq_lengths in test_seq_lengths_list])
    tr_loop_range = list(range(0, train_size, batch_size))
    de_loop_range = list(range(0, dev_size, batch_size))
    test_loop_range = list(range(0, test_size, batch_size))

    last_dev_loss = 0.0
    early_step = 0

    model.train()
    for epoch in range(epoch_num):
        epoch_train_loss = 0.0
        print('Epoch {} starts '.format(epoch))

        ## training
        random.shuffle(tr_loop_range)
        for i_batch in tr_loop_range:

            model_opt.optimizer.zero_grad()
            batch_train_loss = 0.0
            for i in range(process_num):

                batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
                    util.get_batch(batch_size, i_batch, model.process_dim[i], train_seq_lengths_list[i], \
                                    train_seq_times_list[i], train_seq_types_list[i], rnn=False)

                batch_seq_types = batch_seq_types[:, 1:]

                masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim[i], device=device)# exclude the first added even
                model.forward(i, batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
                nll = model.compute_loss(i, batch_seq_times, batch_onehot)

                # 正则项
                # cos_sim = reg.sim_matrix(model.type_emb_list[0].weight, model.type_emb_list[1].weight)
                # ot_loss = torch.sum(reg.sinkhorn2(cos_sim, tau=1e-3, num=100))
                # print('NLL loss: {}, OT loss: {}'.format(nll.data, ot_loss.data))
                # batch_train_loss += nll + 100 * ot_loss

                # emb_list = [model.type_emb_list[0].weight, model.type_emb_list[1].weight]
                # cl_loss = contrastive_loss.cl_loss(emb_list, train_pairs, neg_samples=5, gamma=10)
                # print('NLL loss: {}, CL loss: {}'.format(nll.data, cl_loss.data))
                # batch_train_loss += nll + 10 * cl_loss
                
                batch_train_loss += nll

            batch_train_loss.backward()
            model_opt.optimizer.step()

            if i_batch %50 == 0:
                batch_event_num = torch.sum(batch_seq_lengths).float()
                print('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
                      .format(epoch, i_batch, batch_train_loss.item()/ (batch_event_num * process_num)))
            epoch_train_loss += batch_train_loss.detach()

        if epoch_train_loss < 0:
            break
        train_event_num = sum([torch.sum(train_seq_lengths).float() for train_seq_lengths in train_seq_lengths_list])
        print('---\nEpoch.{} Training set\nTrain Negative Log-Likelihood per event: {:5f} nats\n' \
              .format(epoch, epoch_train_loss / train_event_num))

        ## dev
        dev_event_num, epoch_dev_loss = eval_sahp(batch_size, de_loop_range, dev_seq_lengths_list, dev_seq_times_list,
                                                 dev_seq_types_list, model, device, args.lambda_l2)
        print('Epoch.{} Devlopment set\nDev Negative Likelihood per event: {:5f} nats.\n'. \
              format(epoch, epoch_dev_loss / dev_event_num))

        ## test
        test_event_num, epoch_test_loss = eval_sahp(batch_size, test_loop_range, test_seq_lengths_list, test_seq_times_list,
                                                   test_seq_types_list, model, device, args.lambda_l2)
        print('Epoch.{} Test set\nTest Negative Likelihood per event: {:5f} nats.\n'. \
              format(epoch, epoch_test_loss / test_event_num))

        ## early stopping
        gap = epoch_dev_loss / dev_event_num - last_dev_loss
        if abs(gap) < args.early_stop_threshold:
            early_step += 1
        last_dev_loss = epoch_dev_loss / dev_event_num

        if early_step >=3:
            print('Early Stopping')
            break

        # prediction
        avg_rmse, types_predict_score = \
            prediction_evaluation(device, model, test_seq_lengths_list, test_seq_times_list, 
                                test_seq_types_list, test_size, tmax_list)

    return model


def prediction_evaluation(device, model, test_seq_lengths_list, test_seq_times_list, test_seq_types_list, 
                        test_size, tmax_list):
    model.eval()
    from utils import evaluation
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    process_num = len(test_seq_lengths_list)
    avg_rmse_list = []
    types_predict_score_list = []
    for i in range(process_num):
        test_data = (test_seq_times_list[i], test_seq_types_list[i], test_seq_lengths_list[i])
        incr_estimates, incr_errors, types_real, types_estimates = \
            evaluation.predict_test(model, i, *test_data, pad=model.process_dim[i], device=device,
                                    hmax=tmax_list[i], use_jupyter=False, rnn=False)
        if device != 'cpu':
            incr_errors = [incr_err.item() for incr_err in incr_errors]
            types_real = [types_rl.item() for types_rl in types_real]
            types_estimates = [types_esti.item() for types_esti in types_estimates]

        avg_rmse = np.sqrt(np.mean(incr_errors), dtype=np.float64)
        print("---\nProcess.{} rmse {}".format(i, avg_rmse))
        mse_var = np.var(incr_errors, dtype=np.float64)

        delta_meth_stderr = 1 / test_size * mse_var / (4 * avg_rmse)

        types_predict_score = f1_score(types_real, types_estimates, average='micro')# preferable in class imbalance
        print("Process.{} Type prediction score: {}".format(i, types_predict_score))
        # print("Confusion matrix:\n", confusion_matrix(types_real, types_estimates))

        avg_rmse_list.append(avg_rmse)
        types_predict_score_list.append(types_predict_score)
    model.train()
    return sum(avg_rmse_list) / process_num, sum(types_predict_score_list) / process_num

if __name__ == "__main__":
    mode = 'train'

    if mode == 'train':
        with autograd.detect_anomaly():
            train_eval_sahp()

    else:
        pass
    print("Done!")



