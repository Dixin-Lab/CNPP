import networkx as nx
import numpy as np


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        B_perm (np.ndarray): [d, d] binary adj matrix of DAG with permuted nodes
        P (np.ndarray): [d, d] permutation matrix
    """

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = nx.gnm_random_graph(n=d, m=s0)
        B_und = nx.to_numpy_array(G_und)
        B = np.tril(B_und, k=-1)
    elif graph_type=='BA':
        G_und = nx.barabasi_albert_graph(n=d,m=s0//d)
        B_und = nx.to_numpy_array(G_und)
        B = np.tril(B_und, k=-1)
    else:
        raise ValueError('unknown graph type')
    P = np.random.permutation(np.eye(B_und.shape[0]))
    B_und_perm = P.T @ B_und @ P
    B_perm = P.T @ B @ P
    return B_und, B_und_perm, B, B_perm, P


def simulate_parameter(B, w_ranges=[(0.5, 2.0)]):
    """Simulate SEM parameters for a DAG.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (list): disjoint weight ranges
    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W