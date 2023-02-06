import numpy as np
import torch
import torch.utils.data
from torch.distributions import categorical, exponential, uniform
from typing import Dict, List, Tuple
from constant import PAD
import networkx as nx
import pickle
import scipy.sparse as sp
from networkx import erdos_renyi_graph


# simulator
def exp_kernel(dt: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    gt = w * torch.exp(-w * dt)
    gt[dt < 0] = 0
    return gt


def _conditional_intensity_vector(seq: Dict,
                                  t: torch.Tensor,
                                  infect: torch.Tensor,
                                  mu: torch.Tensor,
                                  w: float = 1.0) -> torch.Tensor:
    lambda_t = torch.clone(mu)
    if seq['ti'] is not None:
        dt = t - seq['ti']
        gt = exp_kernel(dt, w)
        lambda_t += torch.sum(infect[:, seq['ci']] * gt, dim=1)
    return lambda_t


def simulate_ogata_thinning(mu: torch.Tensor,
                            infect: torch.Tensor,
                            num_seq: int,
                            max_time: float = 5.0,
                            w: float = 1.0) -> Tuple[List, List]:
    seqs = []
    seqs_len = []
    for n in range(num_seq):
        sequence = {'ti': None,  # event_times
                    'ci': None,  # event_type
                    'time_since_last_event': None}

        t = 0.0
        lambda_t = torch.clone(mu)
        lambda_all = torch.sum(lambda_t)

        exp_dist = exponential.Exponential(rate=lambda_all)
        unif_dist = uniform.Uniform(low=0.0, high=1.0)

        duration = 0.0
        while t < max_time:
            s = exp_dist.sample()
            u = unif_dist.sample()
            lambda_ts = _conditional_intensity_vector(seq=sequence, t=t + s, infect=infect, mu=mu, w=w)
            lambda_ts_all = torch.sum(lambda_ts)
            lambda_normal = lambda_ts / lambda_ts_all

            t += s
            duration += s

            if t < max_time and u < lambda_ts_all / lambda_all:
                cat_dist = categorical.Categorical(probs=lambda_normal)
                c = cat_dist.sample()
                c = torch.LongTensor([c])
                t_tensor = torch.Tensor([t])
                duration_tensor = torch.Tensor([duration])
                if sequence['ti'] is None:
                    sequence['ti'] = t_tensor
                    sequence['ci'] = c
                    sequence['time_since_last_event'] = duration_tensor
                    # print(sequence['time_since_last_event'],t_tensor)
                else:
                    sequence['ti'] = torch.cat([sequence['ti'], t_tensor], dim=0)
                    sequence['ci'] = torch.cat([sequence['ci'], c], dim=0)
                    sequence['time_since_last_event'] = torch.cat([sequence['time_since_last_event'], duration_tensor],
                                                                  dim=0)

                duration = torch.zeros(1)

            exp_dist = exponential.Exponential(rate=lambda_ts_all)

        if sequence['ci'] is not None:
            print(len(sequence['ci']))

            seqs_len.append(len(sequence['ci']))
            seqs.append(sequence)

    print('Seqs_len: ', seqs_len)
    print('Seqs: ', seqs)
    return seqs, seqs_len


def make_seq(params: Dict, num_seq: int, max_time: float, w: float = 1.0) -> Tuple[List, List]:
    sequences, seqs_len = simulate_ogata_thinning(params['mu'], params['A'], num_seq, max_time, w)
    return sequences, seqs_len


def synthetic_hawkes_parameters(dim: int = 10, thres: float = 0.5) -> Tuple[Dict, Dict, torch.Tensor]:
    infect1 = torch.rand(dim, dim)
    infect1[infect1 < thres] = 0

    mu1 = torch.sum(infect1, dim=0) / torch.sum(infect1)

    print("torch.sum(infect1)", torch.sum(infect1))
    infect1 = 0.8 * infect1 / torch.linalg.svdvals(infect1)[0]
    idx = torch.randperm(dim)
    mu2 = mu1[idx]
    infect2 = infect1[idx, :]
    infect2 = infect2[:, idx]
    pmat = torch.eye(dim)
    pmat = pmat[:, idx]
    params1 = {'mu': mu1, 'A': infect1}
    params2 = {'mu': mu2, 'A': infect2}
    return params1, params2, pmat


def get_synthetic_graph(dim: int = 10, thres: float = 0.5):
    ER = erdos_renyi_graph(dim, thres, seed=None, directed=False)
    infect1 = nx.to_numpy_array(ER)
    print("infect1", infect1)
    print(np.sum(infect1))

    idx = np.random.permutation(dim)
    infect2 = infect1[idx, :]
    infect2 = infect2[:, idx]

    pmat = np.eye(dim, dtype=int)
    pmat = pmat[:, idx]

    edge1 = np.argwhere(infect1).T
    edge2 = np.argwhere(infect2).T

    gnd = np.argwhere(pmat).astype(np.int32)
    np.savez("ER" + str(dim) + ".npz", edge_index1=edge1, edge_index2=edge2, gnd=gnd)


def generate_synthetic_tpps(dim: int = 10,
                            num_seq: int = 200,
                            max_time: float = 30,
                            thres: float = 0.5,
                            w: float = 0.1) -> Tuple[Dict, List, torch.Tensor]:
    params1, params2, pmat = synthetic_hawkes_parameters(dim=dim, thres=thres)
    seqs1, seqs_len1 = make_seq(params=params1, num_seq=num_seq, max_time=max_time, w=w)
    seqs2, seqs_len2 = make_seq(params=params2, num_seq=num_seq, max_time=max_time, w=w)

    return {'seq1': seqs1, 'seq2': seqs2}, [params1, params2], pmat

    # return {'train': [seqs_train, seqs_len_train], 'test': [seqs_test, seqs_len_test]}, [params1, params2], pmat


def generate_synthetic_tpps_from_graph_data(params1, params2,
                                            num_seq: int = 200,
                                            max_time: float = 30,
                                            w: float = 0.1) -> Dict:
    seqs1, seqs_len1 = make_seq(params=params1, num_seq=num_seq, max_time=max_time, w=w)
    seqs2, seqs_len2 = make_seq(params=params2, num_seq=num_seq, max_time=max_time, w=w)

    return {'seq1': seqs1, 'seq2': seqs2}


def save_pkl(data_dict, params1, params2, pmat, max_time, seq_num, output_dir='.', index=1):
    data = {}
    data['params1'] = params1
    data['params2'] = params2
    data['pmat'] = pmat
    data['seq1'] = data_dict['seq1']
    data['seq2'] = data_dict['seq2']

    num1 = pmat.shape[0]
    num2 = pmat.shape[1]

    with open('{}/exp_{}_{}_{}_{}_idx{}.pkl'.format(output_dir, num1, num2, seq_num, max_time, index), 'wb') as f:
        pickle.dump(data, f)
    return


def prepare_dataloader(path):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f)

            num_types = [data['pmat'].shape[0], data['pmat'].shape[1]]

            return data, num_types

    # print('[Info] Loading train data...')
    data, num_types = load_data(path)
    params1 = data['params1']
    params2 = data['params2']
    pmat = data['pmat']

    return params1, params2, pmat


def get_hp_from_syn_npz(zip_path, max_time, num_seq, index, output_dir, decay=0.1, mu_type='max', weighted=False):
    print(zip_path, max_time)
    data = np.load(zip_path)
    mu1 = None
    mu2 = None

    row = data['edge_index1'].T[:, 0]
    col = data['edge_index1'].T[:, 1]
    print(row)
    print(col)
    # no isolated
    n1 = 1 + np.max(data['edge_index1'])
    A1 = sp.coo_matrix((np.ones(data['edge_index1'].shape[1]), (row, col)), shape=(n1, n1)).toarray()
    A1 = torch.from_numpy(A1)

    if weighted:
        A1_w = torch.rand(n1, n1)
        A1_w.uniform_(0.5,1)
        A1_w[A1 == 0] = 0
        A1 = A1_w

    if mu_type == 'max':
        mu1 = torch.sum(A1, dim=0) / torch.max(torch.sum(A1, dim=0))
    elif mu_type == 'sum':
        mu1 = (n1 / 2) * torch.sum(A1, dim=0) / torch.sum(A1)

    A1 = 0.8 * A1 / torch.linalg.svdvals(A1)[0]

    row = data['edge_index2'].T[:, 0]
    col = data['edge_index2'].T[:, 1]
    n2 = 1 + np.max(data['edge_index2'])
    A2 = sp.coo_matrix((np.ones(data['edge_index2'].shape[1]), (row, col)), shape=(n2, n2)).toarray()

    if weighted:
        row = data['gnd'][:, 0]
        col = data['gnd'][:, 1]
        pmat = sp.coo_matrix((np.ones(n2), (row, col)), shape=(n2, n2)).toarray()
        pmat = torch.from_numpy(pmat).float()

        A2 = (pmat.T @ A1 @ pmat).numpy()

    A2 = torch.from_numpy(A2)

    if mu_type == 'max':
        mu2 = torch.sum(A2, dim=0) / torch.max(torch.sum(A2, dim=0))
    elif mu_type == 'sum':
        mu2 = (n2 / 2) * torch.sum(A2, dim=0) / torch.sum(A2)
    if not weighted:
        A2 = 0.8 * A2 / torch.linalg.svdvals(A2)[0]

    params1 = {'mu': mu1, 'A': A1}
    params2 = {'mu': mu2, 'A': A2}

    row = data['gnd'][:, 0]
    col = data['gnd'][:, 1]
    pmat = sp.coo_matrix((np.ones(data['gnd'].shape[0]), (row, col)), shape=(n1, n2)).toarray()
    pmat = torch.from_numpy(pmat)

    print("mu1", mu1)
    print("mu2", mu2)
    print("A1", A1)
    print("A2", A2)

    data_dict = generate_synthetic_tpps_from_graph_data(params1, params2,
                                                        num_seq=num_seq, max_time=max_time,
                                                        w=decay)
    save_pkl(data_dict, params1, params2, pmat, max_time, num_seq, output_dir=output_dir, index=index)


def get_hp_from_exp_pkl(path, max_time, num_seq=10000
                        , index=1, output_dir='.', decay=0.1):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    params1 = data['params1']
    params2 = data['params2']
    pmat = data['pmat']
    data_dict = generate_synthetic_tpps_from_graph_data(params1, params2,
                                                        num_seq=num_seq, max_time=max_time,
                                                        w=decay)
    save_pkl(data_dict, params1, params2, pmat, max_time, num_seq, output_dir=output_dir, index=index)


def get_CoraGraphDataset():
    import dgl.data
    import torch
    import random
    dataset = dgl.data.CoraGraphDataset()
    data = dataset[0]
    adj = data.adj().to_dense()
    dim = adj.shape[0]

    print(torch.sum(adj))
    edge = torch.argwhere(adj)
    non_edge = (adj == 0).nonzero()

    edge_dir_list = []
    for item in edge:
        if item[0] > item[1]:
            edge_dir_list.append(item)

    non_edge_dir_list = []
    for item in non_edge:
        if item[0] > item[1]:
            non_edge_dir_list.append(item)

    random.shuffle(non_edge_dir_list)
    random.shuffle(edge_dir_list)

    edge_dir_list_1 = edge_dir_list + non_edge_dir_list[:528]
    edge_dir_list_2 = edge_dir_list[:4547]

    A1 = torch.zeros((2708, 2708))
    A2 = torch.zeros((2708, 2708))

    for item in edge_dir_list_1:
        i, o = item[0], item[1]
        A1[i, o] = 1
        A1[o, i] = 1

    for item in edge_dir_list_2:
        i, o = item[0], item[1]
        A2[i, o] = 1
        A2[o, i] = 1

    idx = torch.randperm(dim)

    # infect1=torch.tril(infect1)
    A2 = A2[idx, :]
    A2 = A2[:, idx]

    pmat = torch.eye(dim)
    pmat = pmat[:, idx]

    edge1 = torch.argwhere(A1).T.numpy()
    edge2 = torch.argwhere(A2).T.numpy()
    print(edge1.shape[1])
    print(edge2.shape[1])

    gnd = torch.argwhere(pmat)

    np.savez("Cora2708" + ".npz", edge_index1=edge1, edge_index2=edge2, gnd=gnd)


if __name__ == '__main__':  # 30 45 10

    # get_hp_from_syn_npz(zip_path='Tpp_data/Syn_data/ER100.npz', max_time=3, num_seq=2000
    #                     , index=0, output_dir='.'
    #                     , decay=0.1, mu_type='max', weighted=True)

    for path in ['exp_10_10_2000_15_idx0.pkl'
        ,'exp_50_50_2000_6_idx0.pkl','exp_100_100_2000_3_idx0.pkl']:
        for i in range(1,5):
            if path == "exp_10_10_2000_15_idx0.pkl":
                max_time=15
            elif path == 'exp_50_50_2000_6_idx0.pkl':
                max_time=6
            else:
                max_time=3
            get_hp_from_exp_pkl(path=path, max_time=max_time, num_seq=2000
                                    , index=i, output_dir='.', decay=0.1)

    # for path in ['Cora2708.npz','phone-email_0.1.npz','Arenas_noise0.05_0.1.npz']:
    #     for i in range(5):
    #         if path == "Cora2708.npz":
    #             max_time=2.3
    #         elif path == 'phone-email_0.1.npz':
    #             max_time=0.5
    #         else:
    #             max_time=1
    #         print(path)
    #         get_params_from_syn_npz(zip_path=path, max_time=max_time, num_seq=10000
    #                                 , index=i, output_dir='.', decay=0.1, mu_type='max')

    for path in ['Cora2708.npz','phone-email_0.1.npz','Arenas_noise0.05_0.1.npz']:
        for i in range(1,5):
            if path == "Cora2708.npz":
                max_time=2.3
            elif path == 'phone-email_0.1.npz':
                max_time=0.5
            else:
                max_time=1
            print(path)
            get_hp_from_syn_npz(zip_path=path, max_time=max_time, num_seq=10000
                                    , index=i, output_dir='.', decay=0.1, mu_type='max')