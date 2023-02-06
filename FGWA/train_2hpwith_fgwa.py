
from loss import get_non_pad_mask
from HPwithFGWA import HPwithFGWA

import argparse
import numpy as np
import pickle

import torch

import torch.optim as optim

from tqdm import tqdm

from data_util import get_dataloader



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




def main(path0,max_time,idx=1,log_dir='.'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default=path0)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-batch_size', type=int, default=16)
    opt = parser.parse_known_args()[0]

    # default device is CUDA
    opt.device = torch.device('cuda')


    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    valloader_0, valloader_1, \
    trainloader_0, trainloader_1, \
    testloader_0, testloader_1, \
    num_types, params1, params2, pmat = prepare_dataloader(opt)

    training_data_list = [trainloader_0, trainloader_1]
    # validation_data_list = [valloader_0, valloader_1]
    # test_data_list = [testloader_0, testloader_1]

    """ read P_prior """
    #P_prior = torch.from_numpy(np.load(path1)["P"]).to(opt.device).float()
    gnd_pair = np.argwhere(pmat.numpy()).astype(np.int32)

    model = HPwithFGWA(num_type_list=num_types
                       , adj_list=[params1['A'].to(torch.float32),params2['A'].to(torch.float32)]
                       , device=opt.device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)


    t0, t1 = torch.tensor([0]).to(opt.device), torch.tensor([max_time]).to(opt.device)


    topk_dic={1:[],3:[],5:[],10:[],30:[],50:[],100:[]}
    with open('{}FGWA_points_{}_Hitk_idx_{}.pkl'.format(log_dir,str(num_types[0]),idx), 'wb') as f:
        pickle.dump(topk_dic, f)
    for epoch in range(opt.epoch):

        training_data_iters = [iter(dataloader) for dataloader in training_data_list]
        training_data_batch_nums = min([len(dataloader) for dataloader in training_data_list])
        print(training_data_batch_nums)

        #epoch_loss=0


        # for batch_num in range(training_data_batch_nums):
        for batch_num in tqdm(range(training_data_batch_nums), mininterval=2, desc='  - (Training)   ', leave=False):


            """ forward """
            optimizer.zero_grad()
            loss=None
            for process_idx in range(2):

                # if process_idx == 0:
                #     continue

                batch = next(training_data_iters[process_idx])
                """ prepare data """
                event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

                #total_num_event = event_type.ne(PAD).sum().item()


                optimizer.zero_grad()
                input_mask = get_non_pad_mask(event_type)
                # loglik = model(events_batched, input_masks, torch.tensor([0]).cuda(), torch.tensor([end_time]).cuda())

                if process_idx == 0:
                    (lbda, compensator) = model.hp0(event_time,
                                                    event_type, input_mask, t0, t1)
                    loglik = lbda - compensator
                    loss = loglik.mean() * (-1)#/ total_num_event
                else:
                    (lbda, compensator) = model.hp1(event_time,
                                                    event_type, input_mask, t0, t1)
                    loglik = lbda - compensator
                    loss+=loglik.mean() * (-1)

            #print("model.hp1.alpha",model.hp1.alpha)

            fgwa=model.gromov_wasserstein_distance(alpha=0.8)
            loss=loss+100*fgwa
            #print("fgwa",fgwa,"loss",loss)
            loss.backward()

            optimizer.step()

            model.gromov_wasserstein_learning(outer_iteration=100, inner_interation=20, alpha=0.8, tau=0.1)


        #scheduler.step()

        for topk in [1, 3, 5,10 ,30,50,100]:
            # model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy()
            # acc_score_P(model.encoder.event_emb.sinkhorn().T.cpu().detach().numpy(), gnd=gnd_pair[:, ], topk=topk)
            score=acc_score_P(model.trans.cpu().detach().numpy(), gnd=gnd_pair[:, ], topk=topk)
            topk_dic[topk].append(score)
            print("topk",topk,topk_dic[topk])

    ################
    with open('{}FGWA_points_{}_Hitk_idx_{}.pkl'.format(log_dir,str(num_types[0]),idx), 'wb') as f:
        pickle.dump(topk_dic, f)









if __name__=="__main__":

    for path in [("./Tpp_data/Syn_data/unweighted/exp/exp_100_100_2000_3_idx",3
          ,'lambda1_lambda2_const_100_2000_3_'
          ,"./Tpp_data/Syn_data/unweighted/prior/P_100_prior_0.0001_10000_2000_3_idx_")
                 ,("./Tpp_data/Syn_data/unweighted/exp/exp_10_10_2000_15_idx",15
            ,'lambda1_lambda2_const_10_2000_15_'
            ,"./Tpp_data/Syn_data/unweighted/prior/P_10_prior_0.0001_10000_2000_15_idx_")
                 ,("./Tpp_data/Syn_data/unweighted/exp/exp_50_50_2000_6_idx",6
            ,'lambda1_lambda2_const_50_2000_6_'
            ,"./Tpp_data/Syn_data/unweighted/prior/P_50_prior_0.0001_10000_2000_6_idx_")]:

        for idx in range(5):

            plk_path = path[0] + str(idx) + ".pkl"
            Ppath = path[3] + str(idx) + ".npz"
            main(path0=plk_path
                 ,max_time=path[1],idx=idx,log_dir='Log/Syn_data_log/FGWA/')

