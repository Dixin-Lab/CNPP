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
from utils import log_likelihood,type_loss,time_loss
from  model import Transformer



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

    total_data0 = data['seq1']
    total_data1 = data['seq2']

    len0 = len(total_data0)
    len1 = len(total_data1)

    train_len0 = int(len0 * train_ratio)
    test_len0 = int(len0 * (1 - train_ratio) / 2)

    train_len1 = int(len1 * train_ratio)
    test_len1 = int(len1 * (1 - train_ratio) / 2)

    trainloader_0 = get_dataloader(data=total_data0[:train_len0], batch_size=opt.batch_size)
    trainloader_1 = get_dataloader(data=total_data1[:train_len1], batch_size=opt.batch_size)
    testloader_0 = get_dataloader(data=total_data0[train_len0:test_len0 + train_len0], batch_size=opt.batch_size)
    testloader_1 = get_dataloader(data=total_data1[train_len1:test_len1 + train_len1], batch_size=opt.batch_size)

    valloader_0 = get_dataloader(data=total_data0[test_len0 + train_len0:], batch_size=opt.batch_size)
    valloader_1 = get_dataloader(data=total_data1[test_len1 + train_len1:], batch_size=opt.batch_size)


    return valloader_0, valloader_1, trainloader_0, trainloader_1, testloader_0, testloader_1, num_types, params1, params2, pmat


def train_epoch(model, training_data_list, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_event_ll_0 = 0  # cumulative event log-likelihood
    total_time_se_0 = 0  # cumulative time prediction squared-error
    total_event_rate_0 = 0  # cumulative number of correct prediction
    total_num_event_0 = 0  # number of total events
    total_num_pred_0 = 0  # number of predictions

    total_event_ll_1 = 0  # cumulative event log-likelihood
    total_time_se_1 = 0  # cumulative time prediction squared-error
    total_event_rate_1 = 0  # cumulative number of correct prediction
    total_num_event_1 = 0  # number of total events
    total_num_pred_1 = 0  # number of predictions
    # training_data_list: [dataloader0, dataloader1]
    # if process_idx == 1:     event_type -= model.num_types[0] !!!

    training_data_iters = [iter(dataloader) for dataloader in training_data_list]

    training_data_batch_nums = min([len(dataloader) for dataloader in training_data_list])

    print(training_data_batch_nums)

    # for batch_num in range(training_data_batch_nums):
    for batch_num in tqdm(range(training_data_batch_nums), mininterval=2, desc='  - (Training)   ', leave=False):

        batch_num_event = 0  # batch_num_event
        batch_num_pred = 0  # batch_num_pred

        batch_event_loss = None
        batch_time_loss = None
        batch_pred_loss = None

        """ forward """
        optimizer.zero_grad()

        for process_idx in range(2):

            batch = next(training_data_iters[process_idx])
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ process_idx == 1 """
            # if process_idx == 1:
            #     event_type[event_type > 0] -= model.num_types[0]
            # """ forward """
            # optimizer.zero_grad()

            enc_out, prediction = model(process_idx, event_type, event_time)

            """ backward """
            # negative log-likelihood
            event_ll, non_event_ll = log_likelihood(model, process_idx, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)

            # type prediction
            pred_loss, pred_num_event, _, _ = type_loss(prediction[0], event_type, pred_loss_func[process_idx])

            # time prediction
            se = time_loss(prediction[1], event_time)

            # SE is usually large, scale it to stabilize training
            scale_time_loss = event_type.ne(PAD).sum().item() - event_time.shape[0]
            scale_nll_loss = event_type.ne(PAD).sum().item()


            # 3 kinds of loss
            if batch_event_loss is None:
                batch_event_loss = event_loss
                batch_pred_loss = pred_loss
                batch_time_loss = se
            else:
                batch_event_loss += event_loss
                batch_pred_loss += pred_loss
                batch_time_loss += se

            batch_num_event += scale_nll_loss  # batch_num_event
            batch_num_pred += scale_time_loss  # batch_num_pred

            # """ update parameters """
            # optimizer.step()

            # print("train process_idx", process_idx, "acc",
            #      pred_num_event.item() / (event_type.ne(constant.PAD).sum().item() - event_time.shape[0]))

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num_event.item()
            total_num_event += event_type.ne(PAD).sum().item()
            # we do not predict the first event
            total_num_pred += event_type.ne(PAD).sum().item() - event_time.shape[0]

            """ note keeping process_idx """
            if process_idx == 0:
                total_event_ll_0 += -event_loss.item()
                total_time_se_0 += se.item()
                total_event_rate_0 += pred_num_event.item()
                total_num_event_0 += event_type.ne(PAD).sum().item()
                # we do not predict the first event
                total_num_pred_0 += event_type.ne(PAD).sum().item() - event_time.shape[0]
            else:
                total_event_ll_1 += -event_loss.item()
                total_time_se_1 += se.item()
                total_event_rate_1 += pred_num_event.item()
                total_num_event_1 += event_type.ne(PAD).sum().item()
                # we do not predict the first event
                total_num_pred_1 += event_type.ne(PAD).sum().item() - event_time.shape[0]



        if opt.isKL :
            trans = model.encoder.event_emb.sinkhorn().T
            trans = trans / torch.sum(trans)

            trans[trans == 0] = 1

            KL_div = nn.KLDivLoss(reduction="sum")

            ot_loss = KL_div(trans.log(), model.encoder.event_emb.P)

            loss = batch_event_loss / batch_num_event + batch_pred_loss / batch_num_pred \
                   + batch_time_loss / batch_num_pred  + ot_loss*opt.KL_tau
        else:

            loss = batch_event_loss / batch_num_event + batch_pred_loss / batch_num_pred \
                   + batch_time_loss / batch_num_pred

            # loss = batch_event_loss / batch_num_event + batch_pred_loss / batch_num_pred + batch_time_loss / batch_num_pred

        loss.backward()

        """ update parameters """
        optimizer.step()

    rmse_0 = np.sqrt(total_time_se_0 / total_num_pred_0)
    rmse_1 = np.sqrt(total_time_se_1 / total_num_pred_1)
    # rmse = np.sqrt(total_time_se / total_num_pred)
    rmse = (rmse_0 + rmse_1) / 2
    ll0 = total_event_ll_0 / total_num_event_0
    ll1 = total_event_ll_1 / total_num_event_1
    ll = (ll0 + ll1) / 2
    pre0 = total_event_rate_0 / total_num_pred_0
    pre1 = total_event_rate_1 / total_num_pred_1
    pre = (pre0 + pre1) / 2
    print("train epoch process 0", ll0, pre0, rmse_0)
    print("train epoch process 1", ll1, pre1, rmse_1)
    return ll, pre, rmse


# TODO
def eval_epoch(model, validation_data_list, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_event_ll_0 = 0  # cumulative event log-likelihood
    total_time_se_0 = 0  # cumulative time prediction squared-error
    total_event_rate_0 = 0  # cumulative number of correct prediction
    total_num_event_0 = 0  # number of total events
    total_num_pred_0 = 0  # number of predictions

    total_event_ll_1 = 0  # cumulative event log-likelihood
    total_time_se_1 = 0  # cumulative time prediction squared-error
    total_event_rate_1 = 0  # cumulative number of correct prediction
    total_num_event_1 = 0  # number of total events
    total_num_pred_1 = 0  # number of predictions

    true_0 = []
    true_1 = []
    pred_0 = []
    pred_1 = []
    with torch.no_grad():

        # training_data_list: [dataloader0, dataloader1]
        # if process_idx == 1:     event_type -= model.num_types[0] !!!

        validation_data_iters = [iter(dataloader) for dataloader in validation_data_list]

        validation_data_batch_nums = min([len(dataloader) for dataloader in validation_data_list])

        print(validation_data_batch_nums)

        # for batch_num in range(training_data_batch_nums):
        for batch_num in tqdm(range(validation_data_batch_nums), mininterval=2, desc='  - (Training)   ', leave=False):

            for process_idx in range(2):

                batch = next(validation_data_iters[process_idx])
                """ prepare data """
                event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

                """ process_idx == 1 """
                # if process_idx == 1:
                #     event_type[event_type > 0] -= model.num_types[0]

                """ forward """
                enc_out, prediction = model(process_idx, event_type, event_time)

                """ compute loss """
                event_ll, non_event_ll = log_likelihood(model, process_idx, enc_out, event_time, event_type)
                event_loss = -torch.sum(event_ll - non_event_ll)

                _, pred_num, true_list, pred_list = type_loss(prediction[0], event_type, pred_loss_func[process_idx])
                """ f1 score """
                if process_idx == 0:
                    true_0.extend(true_list)
                    pred_0.extend(pred_list)
                else:
                    true_1.extend(true_list)
                    pred_1.extend(pred_list)

                se = time_loss(prediction[1], event_time)

                # print("eval process_idx",process_idx,"acc",pred_num.item()/(event_type.ne(constant.PAD).sum().item() - event_time.shape[0]))
                """ note keeping """
                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_event_rate += pred_num.item()
                total_num_event += event_type.ne(PAD).sum().item()
                total_num_pred += event_type.ne(PAD).sum().item() - event_time.shape[0]

                """ note keeping process_idx """
                if process_idx == 0:
                    total_event_ll_0 += -event_loss.item()
                    total_time_se_0 += se.item()
                    total_event_rate_0 += pred_num.item()
                    total_num_event_0 += event_type.ne(PAD).sum().item()
                    # we do not predict the first event
                    total_num_pred_0 += event_type.ne(PAD).sum().item() - event_time.shape[0]
                else:
                    total_event_ll_1 += -event_loss.item()
                    total_time_se_1 += se.item()
                    total_event_rate_1 += pred_num.item()
                    total_num_event_1 += event_type.ne(PAD).sum().item()
                    # we do not predict the first event
                    total_num_pred_1 += event_type.ne(PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    rmse_0 = np.sqrt(total_time_se_0 / total_num_pred_0)
    rmse_1 = np.sqrt(total_time_se_1 / total_num_pred_1)
    rmse = (rmse_0 + rmse_1) / 2

    ll0 = total_event_ll_0 / total_num_event_0
    ll1 = total_event_ll_1 / total_num_event_1
    ll = (ll0 + ll1) / 2

    pre0 = total_event_rate_0 / total_num_pred_0
    pre1 = total_event_rate_1 / total_num_pred_1
    pre = (pre0 + pre1) / 2

    print("test epoch process 0", ll0, pre0, rmse_0)
    print("test epoch process 1", ll1, pre1, rmse_1)

    from sklearn.metrics import f1_score
    f1_0 = f1_score(true_0, pred_0, average='micro')
    f1_1 = f1_score(true_1, pred_1, average='micro')
    true_1_ = list(map(lambda x: x + model.num_types[0], true_1))
    pred_1_ = list(map(lambda x: x + model.num_types[0], pred_1))
    true_1_.extend(true_0)
    pred_1_.extend(pred_0)
    f1_total = (f1_0 + f1_1) / 2
    print("test epoch process 0, f1 score", f1_0)
    print("test epoch process 1, f1 score", f1_1)
    print("test epoch total, f1 score", f1_total)
    return ll, pre, rmse,f1_total


def train(model, training_data_list, validation_data_list, test_data_list, optimizer, scheduler, pred_loss_func: list,
          opt, gnd_pair):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE

    test_event_losses = []  # validation log-likelihood
    test_pred_losses = []  # validation event type prediction accuracy
    test_rmse = []  # validation event time prediction RMSE

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data_list, optimizer, pred_loss_func, opt)
        print('  -Total (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time,val_f1 = eval_epoch(model, validation_data_list, pred_loss_func, opt)
        print('  -Total (Valid)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Val Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 12.8f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        test_event, test_type, test_time,test_f1 = eval_epoch(model, test_data_list, pred_loss_func, opt)
        print('  -Total (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60))

        test_event_losses += [test_event]
        test_pred_losses += [test_type]
        test_rmse += [test_time]
        print('  - [Test Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(test_event_losses), pred=max(test_pred_losses), rmse=min(test_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('val {epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {f1: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time,f1=val_f1))

            f.write('test {epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {f1: 8.5f}\n'
                    .format(epoch=epoch, ll=test_event, acc=test_type, rmse=test_time,f1=test_f1))
        # X1 = model.encoder.event_emb_list[0].weight.cpu().detach()[1:].numpy()
        # X2 = model.encoder.event_emb_list[1].weight.cpu().detach()[1:].numpy()
        # from sklearn.metrics.pairwise import cosine_similarity
        # cosine_dis = cosine_similarity(X1, X2)
        # from Joint_sahp.metric import acc_score_P
        for topk in [1, 3, 5,10 ,30,50,100]:
            # model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy()
            # acc_score_P(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy(), gnd=gnd_pair[:, ], topk=topk)
            score= acc_score_P(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy(), gnd=gnd_pair[:, ], topk=topk)

            with open(opt.log, 'a') as f:
                f.write('top {top}, {score: 8.5f}\n'
                        .format(top=topk,score=score))


        scheduler.step()


def main(path0, path1,log_path,tau,isKL=True,is_param=False):
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default=path0)

    parser.add_argument('-epoch', type=int, default=50)
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

    parser.add_argument('-log', type=str, default=log_path)

    parser.add_argument('-isKL', type=bool, default=isKL)
    parser.add_argument('-is_param', type=bool, default=is_param)
    parser.add_argument('-KL_tau', type=float, default=tau)
    opt = parser.parse_known_args()[0]

    # default device is CUDA
    opt.device = torch.device('cuda')
    # torch.device('cuda')
    # [(128,256),(32,64)]
    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

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
    P_prior = torch.from_numpy(np.load(path1)["P"]).to(opt.device).float()

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
    # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                       opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    # TO DO
    if opt.smooth > 0:
        pred_loss_func0 = LabelSmoothingLoss(opt.smooth, num_types[0], ignore_index=-1)
        pred_loss_func1 = LabelSmoothingLoss(opt.smooth, num_types[1], ignore_index=-1)
        pred_loss_func = [pred_loss_func0, pred_loss_func1]
    else:
        pred_loss_func0 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_func1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_func = [pred_loss_func0, pred_loss_func1]

    train(model, training_data_list, validation_data_list, test_data_list, optimizer, scheduler, pred_loss_func, opt,
          gnd_pair)


import random


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
    # print("gnd",gnd)
    for idx in range(length):
        if gnd[idx, right] in possible_alignment[gnd[idx, 1 - right]][-topk:]:
            num += 1
    return num / length


def acc_score_P(P, gnd, topk=5):
    dis = P
    score=topk_alignment_score(dis, gnd, topk, right=1)
    print("topk", topk, "score:",score)
    return score



if __name__ == '__main__':

    # for path in [("./Tpp_data/Syn_data/unweighted/exp/exp_100_100_2000_3_idx", 3
    #               , 'lambda1_lambda2_const_100_2000_3_'
    #               , "./Tpp_data/Syn_data/unweighted/prior/P_100_prior_0.0001_10000_2000_3_idx_")
    #     , ("./Tpp_data/Syn_data/unweighted/exp/exp_10_10_2000_15_idx", 15
    #        , 'lambda1_lambda2_const_10_2000_15_'
    #        , "./Tpp_data/Syn_data/unweighted/prior/P_10_prior_0.0001_10000_2000_15_idx_")
    #     , ("./Tpp_data/Syn_data/unweighted/exp/exp_50_50_2000_6_idx", 6
    #        , 'lambda1_lambda2_const_50_2000_6_'
    #        , "./Tpp_data/Syn_data/unweighted/prior/P_50_prior_0.0001_10000_2000_6_idx_")]:
    #
    #     for idx in range(5):
    #         plk_path = path[0] + str(idx) + ".pkl"
    #         log_path="./Log/Syn_data_log/parameter/parameter_with_KL/"+path[2][22:]+"idx"+str(idx)+"_tau"+str(tau)+".txt"
    #         Ppath = path[3] + str(idx) + ".npz"
    #         main(plk_path, Ppath ,log_path=log_path,tau=1,isKL=True,is_param=False
    #              )


    path=("./Tpp_data/Real_data/exp/exp_50_50_2000_6_idx", 6
           , 'lambda1_lambda2_const_50_2000_6_'
           , "./Tpp_data/Real_data/prior/P_50_prior_0.0001_10000_2000_6_idx_")


    for is_param in [True,False]:
        for tau in [0.01,0,1,100]:
            for idx in range(5):
                plk_path = path[0] + str(idx) + ".pkl"

                if is_param:
                    log_path = "./Log/Real_data_log/parameter/parameter_with_KL/" + path[2][22:] + "idx" + str(
                        idx) + "_tau" + str(tau) + ".txt"

                    Ppath = path[3] + str(idx) + ".npz"
                    main(plk_path, Ppath, log_path=log_path, tau=tau, isKL=True, is_param=is_param
                         )
                else:
                    log_path = "./Log/Real_data_log/non_parameter/non_parameter_with_KL/" + path[2][22:] + "idx" + str(
                        idx) + "_tau" + str(tau) + ".txt"

                    Ppath = path[3] + str(idx) + ".npz"
                    main(plk_path, Ppath, log_path=log_path, tau=tau, isKL=True, is_param=is_param
                         )



        #     nohup python simulator.py >/dev/null 2>log &