import ot
import argparse
import torch
import numpy as np
import pickle

element_max = np.vectorize(max)


def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        num_types = [data['pmat'].shape[0], data['pmat'].shape[1]]

        return data, num_types


def get_prior_P(path0, path1, epsilon=0.1, numItermax=5000):
    data, num_types = load_data(path0)

    a_b_cost = np.load(path1)
    cost = a_b_cost['cost']
    lambda_1 = a_b_cost['lambda_1']

    distribute_a = np.ones(num_types[0]) / num_types[0]
    distribute_b = np.ones(num_types[1]) / num_types[1]

    P = ot.sinkhorn(distribute_a, distribute_b, cost,
                    epsilon, numItermax=numItermax)

    np.savez("P_" + str(len(lambda_1)) + "_prior.npz", P=P)


def get_tpp_statistics(path="."):
    data, num_types = load_data(path)

    seq1 = data['seq1']
    seq2 = data['seq2']

    lambda_1 = torch.zeros(num_types[0])
    lambda_2 = torch.zeros(num_types[1])

    for seq in seq1:
        T = seq['ti'][-1]
        average_intensity_seq = np.zeros(num_types[0])
        for event in seq['ci']:
            average_intensity_seq[int(event)] += 1

        average_intensity_seq /= T
        lambda_1 = lambda_1 + average_intensity_seq

    for seq in seq2:
        T = seq['ti'][-1]
        average_intensity_seq = np.zeros(num_types[1])
        for event in seq['ci']:
            average_intensity_seq[int(event)] += 1

        average_intensity_seq /= T
        lambda_2 = lambda_2 + average_intensity_seq

    lambda_1 /= len(seq1)
    lambda_2 /= len(seq2)

    lambda_1 = lambda_1.unsqueeze(-1).numpy()
    lambda_2 = lambda_2.unsqueeze(-1).numpy()

    from sklearn.metrics.pairwise import euclidean_distances
    cost = euclidean_distances(lambda_1, lambda_2)

    np.savez("lambda1_lambda2_const.npz", lambda_1=lambda_1, lambda_2=lambda_2, cost=cost)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-plk_path', type=str, default='exp_10_10_100_30_idx0.pkl')
    parser.add_argument('-cost_path', type=str, default='lambda1_lambda2_const.npz',
                        help='the cost matrix path')
    parser.add_argument('-max_time', type=float, default=30)
    parser.add_argument('-num_seq', type=int, default=2000)
    parser.add_argument('-numItermax', type=int, default=10000, help='max num iteration for sinkhorn algorithm')
    parser.add_argument('-epsilon', type=int, default=0.1, help='epsilon for sinkhorn algorithm')
    parser.add_argument('-index', type=int, default=0, help='give a index to the prior P')
    parser.add_argument('-out_put', type=str, default='.')

    opt = parser.parse_known_args()[0]

    get_tpp_statistics(path=opt.plk_path)
    get_prior_P(path0=opt.plk_path, path1=opt.cost_path, epsilon=opt.epsilon,
                numItermax=opt.numItermax)


if __name__ == "__main__":
    main()
