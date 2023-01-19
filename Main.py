import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import constant
import Utils

from simulator import get_dataloader
from model import Transformer2,Transformer
from tqdm import tqdm

import matplotlib.pyplot as plt


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
            num_types = [data['pmat'].shape[0],data['pmat'].shape[1]]

            return data, num_types

    # print('[Info] Loading train data...')
    data, num_types = load_data(opt.data)
    params1 = data['params1']
    params2 = data['params2']
    pmat = data['pmat']
    lt=len(data['train'])
    le=len(data['test'])
    trainloader = get_dataloader(data=data['train'],batch_size=opt.batch_size)
    testloader = get_dataloader(data=data['test'], batch_size=opt.batch_size)
    trainloader_0 = get_dataloader(data=data['train'][:lt//2],batch_size=opt.batch_size)
    trainloader_1 = get_dataloader(data=data['train'][lt//2:],batch_size=opt.batch_size)
    testloader_0 = get_dataloader(data=data['test'][:le//2],batch_size=opt.batch_size)
    testloader_1 = get_dataloader(data=data['test'][le//2:],batch_size=opt.batch_size)
    return trainloader, testloader,trainloader_0,trainloader_1,testloader_0,testloader_1, num_types,params1,params2,pmat


def train_epoch_nll(model, training_data, optimizer, opt, P_prior):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_num_event = 0  # number of total events

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        #enc_out, (type_prediction, time_prediction) = model(event_type, event_time)
        enc_out = model(event_type, event_time)
        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        print("event_ll, non_event_ll",event_ll, non_event_ll)
        event_loss = -torch.sum(event_ll - non_event_ll)


        #ot loss,trans matrix KL div
        trans=model.encoder.event_emb.sinkhorn().T
        trans=trans/torch.sum(trans)

        KL_div= nn.KLDivLoss(reduction="sum")

        ot_loss=KL_div(trans.log(),P_prior)


        loss = event_loss+ot_loss*1000 #+pred_loss + se / scale_time_loss


        loss.backward(retain_graph=True)


        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        print("-event_loss.item()",-event_loss.item())
        # total_time_se += se.item()
        # total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(constant.PAD).sum().item()
        # we do not predict the first event
        #total_num_pred += event_type.ne(constant.PAD).sum().item() - event_time.shape[0]




    #rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event  #, total_event_rate / total_num_pred, rmse


def train_epoch_event_and_time(process_idx,model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)


        """ forward """
        optimizer.zero_grad()

        (type_prediction, time_prediction) = model.predict_event_and_time(process_idx, event_type, event_time)

        """ backward """

        # offset
        if process_idx == 1:
            event_type[event_type > 0] -= model.num_types[0]

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(type_prediction, event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(time_prediction, event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100

        #ot loss,trans matrix KL div
        # trans=model.encoder.event_emb.sinkhorn().T
        # trans=trans/torch.sum(trans)
        # KL_div= nn.KLDivLoss(reduction="sum")
        # ot_loss=KL_div(trans.log(),P_prior)


        loss = pred_loss + se / scale_time_loss

        loss.backward(retain_graph=True)

        """ update parameters """
        optimizer.step()

        """ note keeping """

        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(constant.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(constant.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return  total_event_rate / total_num_pred, rmse


def eval_epoch_nll(model, validation_data,opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_num_event = 0  # number of total events
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)


            """ note keeping """
            total_event_ll += -event_loss.item()
            total_num_event += event_type.ne(constant.PAD).sum().item()


    return total_event_ll / total_num_event

def eval_epoch_event_and_time(process_idx,model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()


    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_pred = 0  # number of predictions

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            #print("event_type.shape ",event_type.shape)


            """ forward """
            (type_prediction, time_prediction) = model.predict_event_and_time(process_idx, event_type, event_time)

            """ compute loss """
            # offset
            if process_idx == 1:
                event_type[event_type > 0] -= model.num_types[0]


            _, pred_num = Utils.type_loss(type_prediction, event_type, pred_loss_func)
            se = Utils.time_loss(time_prediction, event_time)

            """ note keeping """
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_pred += event_type.ne(constant.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_rate / total_num_pred, rmse

def train_backbone(model, training_data, validation_data, optimizer, scheduler, P_prior, opt, gnd_pair):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    max_ll=-1e20
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event= train_epoch_nll(model, training_data, optimizer,opt, P_prior)

        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60))

        #save model
        if train_event>max_ll:
            max_ll=train_event
            torch.save(model.state_dict(), 'save.pt')

        start = time.time()
        ###############################最后来
        valid_event= eval_epoch_nll(model, validation_data, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]

        print('  - [Info] Maximum ll: {event: 8.5f}, '
              .format(event=max(valid_event_losses)))



        scheduler.step()
        from Joint_sahp.metric import acc_score_P
        for topk in [1,3,5]:#,30,50
            acc_score_P(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy(), gnd=gnd_pair[:, ], topk=topk)
            #acc_score_P(pmt, gnd=gnd_pair[:, ], topk=topk)

        plt.figure(figsize=(10, 5))
        plt.imshow(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy())
        plt.colorbar()
        plt.savefig('./Plot/sinkhorn_epoch'+str(epoch)+'.pdf')


def train_event_and_time(process_idx,model, training_data, validation_data, optimizer, scheduler, pred_loss_func,opt):
    """ Start training. """


    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    max_acc=0
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_type,train_time= train_epoch_event_and_time(process_idx , model, training_data, optimizer, pred_loss_func, opt)

        print('  - (Training)    accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        ###############################最后来(process_idx,model, validation_data, pred_loss_func, opt)
        valid_type, valid_time = eval_epoch_event_and_time(process_idx,model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        if valid_type>max_acc:
            max_acc=valid_type
            torch.save(model.state_dict(), 'save.pt')

        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(pred=max(valid_pred_losses), rmse=min(valid_rmse)))


        scheduler.step()


def main():
    #with torch.autograd.set_detect_anomaly(True)

    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default="exp_10_10_2000.pkl")

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=12)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')
    #torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader,\
    trainloader_0,trainloader_1,\
    testloader_0,testloader_1,\
    num_types,params1,params2,pmat= prepare_dataloader(opt)

    """ prepare model """
    model = Transformer2(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
    #                        opt.lr, betas=(0.9, 0.999), eps=1e-05)
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func0 = Utils.LabelSmoothingLoss(opt.smooth, num_types[0], ignore_index=-1)
        pred_loss_func1 = Utils.LabelSmoothingLoss(opt.smooth, num_types[1], ignore_index=-1)
    else:
        pred_loss_func0 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_func1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    plt.figure(figsize=(10, 5))
    plt.imshow(pmat)
    plt.colorbar()
    plt.savefig('./Plot/pmat.pdf')
    P=torch.from_numpy(np.load("P_100_prior0.0001.npz")["P"]).to(opt.device).float()
    # P=P.
    gnd_pair=np.argwhere(pmat.numpy()).astype(np.int32)


    #first train nll
    #train_backbone(model, trainloader, testloader, optimizer, scheduler, P, opt, gnd_pair=gnd_pair)
    #0 pred_loss_func0 trainloader_0, testloader_0
    #1 pred_loss_func1  trainloader_1, testloader_1


    #then train event acc and time r
    #model.load_state_dict(torch.load("./save.pt"))

    train_event_and_time(0,model,trainloader_0, testloader_0, optimizer, scheduler, pred_loss_func0,opt)
    train_event_and_time(1, model, trainloader_1, testloader_1, optimizer, scheduler, pred_loss_func1, opt)





import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    #setup_seed(888)
    main()
