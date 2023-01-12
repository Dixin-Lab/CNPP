import os 
import datetime
import numpy as np
from utils import simulate_dag, simulate_parameter
from simulate import make_seq


if __name__ == '__main__':

    num_nodes, num_edges = 10, 80
    graph_type = 'ER'
    w_ranges=[(0.5, 2.0)]
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
    # graph_dict = {'unweighted_directed': [B, B_perm],
    #                 'weighted_directed': [W, W_perm]}

    graph_dict = {#'unweighted_undirected': [B_und, B_und_perm],
                  #'weighted_undirected': [W_und, W_und_perm],
                  #'unweighted_directed': [B, B_perm],
                  'weighted_directed': [W, W_perm]
                  }

    # Use Ogata's thinning method and save sequence data
    num_seq = 100
    max_time = 20
    w = 1.0
    for graph_type, graph_list in graph_dict.items():
        process_num = len(graph_list)
        SIMULATE_PATH = os.path.join(SAVED_MODELS_PATH, graph_type)
        os.makedirs(SIMULATE_PATH, exist_ok=True)
        for i in range(process_num):
            sequences, max_len = make_seq(i, graph_list[i], num_seq, max_time, w, SIMULATE_PATH)

    
    

