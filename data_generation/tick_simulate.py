import matplotlib.pyplot as plt
import numpy as np
from tick.base import TimeFunction
from tick.plot import plot_timefunction
from tick.plot import plot_point_process
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc,HawkesSumExpKern
import pickle
import torch
from torch.distributions import categorical, exponential, uniform
def save_pkl(process_idx, info, seqs, seqs_len, output_dir='.'):
    data = {}
    mu,alpha,decay,tmax = info

    data['mu'] = list(mu)
    data['alpha'] = np.array(alpha)
    data['decay'] = decay
    data['tmax'] = tmax

    data['timestamps'] = tuple(np.array(seq['ti']) for seq in seqs)
    data['types'] = tuple(np.array(seq['ci']) for seq in seqs)
    data['lengths'] = seqs_len

    with open('{}/exp_{}d_{}.pkl'.format(output_dir, mu.shape[0], process_idx), 'wb') as f:
        pickle.dump(data, f)
    return


import torch.nn as nn
def process_loaded_sequences(loaded_hawkes_data: dict, process_dim: int) :
    """
    Preprocess synthetic Hawkes data by padding the sequences.
    Args:
        loaded_hawkes_data:
        process_dim:
        tmax:
    Returns:
        sequence event times, event types and overall lengths (dim0: batch size)
    """
    # Tensor of sequence lengths (with additional BOS event)
    seq_lengths = torch.Tensor(loaded_hawkes_data['lengths']).int()

    event_times_list = loaded_hawkes_data['timestamps']
    event_types_list = loaded_hawkes_data['types']
    event_times_list = [torch.from_numpy(e) for e in event_times_list]
    event_types_list = [torch.from_numpy(e) for e in event_types_list]

    tmax = 0
    for tsr in event_times_list:
        if torch.max(tsr) > tmax:
            tmax = torch.max(tsr)

    #  Build a data tensor by padding
    seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value=tmax).float()
    seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim=1) # add 0 to the sequence beginning

    seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value=process_dim)
    seq_types = torch.cat(
        (process_dim*torch.ones_like(seq_types[:, :1]), seq_types), dim=1).long()# convert from floattensor to longtensor

    return seq_times, seq_types, seq_lengths, tmax


def get_syn_data(run_time = 300):
    T00 = np.arange(0, 4, 0.1)
    Y00 = 0.2 * (0.5 + T00) ** (-1.3)
    Y01 = 0.03 * np.exp(-0.3 * T00)
    Y10 = 0.05 * np.exp(-0.2 * T00) + 0.16 * np.exp(-0.8 * T00)
    Y11 = np.sin(T00) / 8
    Y11[Y11 < 0] = 0

    tf_00 = TimeFunction((T00, Y00), dt=0, border_type=TimeFunction.Border0)
    tf_01 = TimeFunction((T00, Y01), dt=0)
    tf_10 = TimeFunction((T00, Y10), dt=0)
    tf_11 = TimeFunction((T00, Y11), dt=0)


    seqs = []
    seqs_len = []
    for i in range(4000):
        hawkes = SimuHawkes(n_nodes=2, end_time=run_time, verbose=False)
        hawkes.set_kernel(0, 0, HawkesKernelTimeFunc(tf_00))
        hawkes.set_kernel(0, 1, HawkesKernelTimeFunc(tf_01))
        hawkes.set_kernel(1, 0, HawkesKernelTimeFunc(tf_10))
        hawkes.set_kernel(1, 1, HawkesKernelTimeFunc(tf_11))
        hawkes.set_baseline(0, 0.1)
        hawkes.set_baseline(1, 0.2)
        dt = 0.001
        hawkes.track_intensity(dt)
        hawkes.simulate()
        timestamps = hawkes.timestamps
        intensity = hawkes.tracked_intensity
        intensity_times = hawkes.intensity_tracked_times
        # print(timestamps[1])
        sequence = {'ti': None,  # event_times
                    'ci': None,  # evnet_type
                    }
        array0 = timestamps[0]
        array1 = timestamps[1]
        len0 = len(array0)
        len1 = len(array1)
        ti_array = np.hstack((array0, array1))
        ci_array = np.hstack((np.zeros(len0, dtype=int), np.ones(len1, dtype=int)))
        sorted_idx = np.argsort(ti_array)
        ti_array = ti_array[sorted_idx]
        ci_array = ci_array[sorted_idx]
        sequence['ti'] = ti_array
        sequence['ci'] = ci_array
        seqs.append(sequence)
        seqs_len.append(len(ci_array))
        print(i, len(ci_array))

    info=torch.tensor([0.1,0.2]),torch.ones((2,2)),1,300
    save_pkl(process_idx=0, info=info, seqs=seqs, seqs_len=seqs_len, output_dir='./syn_data')




def get_graph_data(process_idx, infect, w, n_nodes, mu, output_dir, max_time = 1000, num_seq = 10):

    from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti

    infect= infect[:, :, np.newaxis]
    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=infect, decays=w, baseline=mu,
        end_time=max_time, verbose=False)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=num_seq)

    multi.end_time = [max_time for i in range(num_seq)]
    multi.simulate()


    seqs = []
    seqs_len = []
    timestamps = multi.timestamps
    for i in range(num_seq):

        sequence = {'ti': None,  # event_times
                    'ci': None,  # evnet_type
                    }
        ti_array = np.empty(0)
        ci_array = np.empty(0)
        for j in range(n_nodes):
            array = timestamps[i][j]
            l = len(array)
            ti_array = np.hstack((ti_array, array))
            ci_array = np.hstack((ci_array,np.full(l,j)))
            sorted_idx = np.argsort(ti_array)
            ti_array = ti_array[sorted_idx]
            ci_array = ci_array[sorted_idx]

        sequence['ti'] = ti_array
        sequence['ci'] = ci_array
        seqs.append(sequence)
        seqs_len.append(len(ci_array))
        print(i, len(ci_array))

    info= mu, infect, w, max_time
    save_pkl(process_idx=process_idx, info=info, seqs=seqs, seqs_len=seqs_len, output_dir=output_dir)

    #https: // x - datainitiative.github.io / tick / modules / generated / tick.hawkes.HawkesSumExpKern.html  # tick.hawkes.HawkesSumExpKern


def make_seq(process_idx, data, num_seq, max_time, w, output_dir='.'):
    mu = np.sum(data, axis=1) / np.sum(data)
    data = torch.Tensor(data)
    event_type = data.shape[0]
    #mu = np.ones(event_type) / event_type

    print("mu",mu)
    # deg = torch.sum(data, dim=1)
    # mu = torch.where(deg > 0, deg, torch.tensor(eps, dtype=torch.float32)) / torch.sum(data)
    # mu = torch.zeros(event_type, dtype=torch.float32)
    # mu[nodes] = 1.0
    infect = data / torch.linalg.svdvals(data)[0]
    # infect = data / torch.sum(data, dim=1, keepdim=True)
    # print('SVD before: ', torch.linalg.svdvals(data)[0])
    # print('SVD after: ', torch.linalg.svdvals(infect)[0])
    # infect = data / 5

    get_graph_data(infect=infect.numpy(), mu=mu, w=[w], process_idx=process_idx, n_nodes=event_type,
                   num_seq=num_seq, max_time=max_time, output_dir=output_dir)


import os
import datetime
from utils import simulate_dag, simulate_parameter

if __name__ == '__main__':

    num_nodes, num_edges = 10, 90
    graph_type = 'ER'
    w_ranges = [(0.5, 2.0)]
    train_ratio = 0.2

    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
    SAVED_MODELS_PATH = '../data/{}_{}_{}'.format(graph_type, num_nodes, num_edges)
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

    # Generate unweighted graph
    B_und, B_und_perm, B, B_perm, P = simulate_dag(num_nodes, num_edges, graph_type)

    gnd_pairs = np.argwhere(P).astype(np.int32)
    num_train = int(num_nodes * train_ratio)
    train_pairs = gnd_pairs[0:int(num_nodes * train_ratio), :]
    test_pairs = gnd_pairs[int(num_nodes * train_ratio):, :]

    # Assign weights for unweighted graph
    W_und = simulate_parameter(B_und, w_ranges=w_ranges)
    W_und_perm = P.T @ W_und @ P
    W = np.tril(W_und, k=-1)
    W_perm = P.T @ W @ P

    file_pairs_name = os.path.join(SAVED_MODELS_PATH, 'pairs.npz')
    np.savez(file_pairs_name, gnd_pairs=gnd_pairs, train_pairs=train_pairs, test_pairs=test_pairs)
    graph_dict = {#'unweighted_directed': [B, B_perm],
                    'weighted_directed': [W, W_perm],
            #'weighted_undirected': [W_und, W_und_perm]
        }

    # graph_dict = {#'unweighted_undirected': [B_und, B_und_perm],
    #               'weighted_undirected': [W_und, W_und_perm]}


    num_seq = 4000
    max_time = 150
    w = 1.0

    for graph_type, graph_list in graph_dict.items():
        process_num = len(graph_list)

        for i in range(process_num):
            SIMULATE_PATH = os.path.join(SAVED_MODELS_PATH, graph_type)
            os.makedirs(SIMULATE_PATH, exist_ok=True)
            make_seq(process_idx=i,data=graph_list[i]
                     ,num_seq=num_seq,max_time=max_time,w=w,output_dir=SIMULATE_PATH)