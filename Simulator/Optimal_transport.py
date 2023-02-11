import ot
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
element_max = np.vectorize(max)


def sinkhorn(C, a, b, epsilon, precision):
    a = a.reshape((C.shape[0], 1))
    b = b.reshape((C.shape[1], 1))
    K = np.exp(-C / epsilon)
    print("a", a)
    print("b", b)
    print("K", K)
    # initialization
    u = np.ones((C.shape[0], 1))
    v = np.ones((C.shape[1], 1))
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    p_norm = np.trace(P.T @ P)
    print("p_norm", p_norm)
    while True:
        u = a / element_max((K @ v), 1e-300)  # avoid divided by zero
        v = b / element_max((K.T @ u), 1e-300)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        print("u", max(u))
        print("v", max(v))
        print("p_norm", p_norm)
        print("np.trace(P.T @ P)", np.trace(P.T @ P))
        print("precision", abs((np.trace(P.T @ P) - p_norm) / p_norm))
        if abs((np.trace(P.T @ P) - p_norm) / p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    return P, np.trace(C.T @ P)


def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        num_types = [data['pmat'].shape[0], data['pmat'].shape[1]]

        return data, num_types


def get_prior_P(path0, path1, epsilon=0.0001, numItermax=5000, max_time=0.0, num_seq=2000, index=1):

    data, num_types = load_data(path0)

    a_b_cost = np.load(path1)
    cost = a_b_cost['cost']
    lambda_1 = a_b_cost['lambda_1']

    print("cost", cost)

    distribute_a = np.ones(num_types[0]) / num_types[0]
    distribute_b = np.ones(num_types[1]) / num_types[1]

    P = ot.sinkhorn(distribute_a, distribute_b, cost,
                    epsilon, numItermax=numItermax)

    np.savez("P_" + str(len(lambda_1)) + "_prior_" + str(epsilon) + "_" +
             str(numItermax)+"_"+str(num_seq)+"_"+str(max_time)+"_idx_"+str(index)+".npz", P=P)
    print("P", sum(sum(P)))
    # plt.figure(figsize=(10, 5))
    # plt.imshow(P)
    # plt.colorbar()
    # # plt.savefig('./Plot/P'+str(epsilon)+'.pdf')
    # plt.show()
    # plt.close()
    #
    # pmat = data['pmat']
    # plt.figure(figsize=(10, 5))
    # plt.imshow(pmat)
    # plt.colorbar()
    # #plt.savefig('./Plot/P_'+str(epsilon)+'_'+str(len(a))+'.pdf')
    # plt.show()
    # plt.close()


def prior_accuracy(path0, path1):
    data, num_types = load_data(path0)
    pmat = data['pmat']

    P_prior = np.load(path1)['P']

    plt.figure()
    plt.imshow(pmat)
    plt.colorbar()
    plt.savefig(path0+"gnd.pdf")
    plt.close()

    plt.figure()
    plt.imshow(P_prior)
    plt.colorbar()
    plt.savefig(path0 + "pred.pdf")
    plt.close()

    print(np.sum(P_prior))
    gnd_pair = np.argwhere(pmat.numpy()).astype(np.int32)
    from metric import acc_score_P
    for topk in [1, 3, 5, 10, 30, 50]:  # ,30,50
        acc_score_P(P_prior, gnd=gnd_pair[:, ], topk=topk)
        # acc_score_P(pmt, gnd=gnd_pair[:, ], topk=topk)


def get_tpp_statistics(path="./exp_5313_5120_10000_0.05.pkl", num_seq=10000, max_time=0.05, index=0):
    data, num_types = load_data(path)

    seq1 = data['seq1']
    seq2 = data['seq2']
    print(len(seq1[1]['ti']))
    print(data['params1'])
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
    # print("lambda_1:", lambda_1)
    # print("lambda_2:", lambda_2)
    # print("cost", cost)
    np.savez("lambda1_lambda2_const_" + str(len(lambda_1)) + "_" + str(num_seq) + "_" +
             str(max_time) + "_idx"+str(index)+".npz", lambda_1=lambda_1, lambda_2=lambda_2, cost=cost)


def get_tpp_statistics_num_seq(path="./exp_5313_5120_10000_0.05.pkl", num_seq=10000, max_time=0.05):
    data, num_types = load_data(path)

    seq1 = data['seq1']
    seq2 = data['seq2']

    lambda_1 = torch.zeros(num_types[0])
    lambda_2 = torch.zeros(num_types[1])

    for seq in seq1[:num_seq]:
        T = seq['ti'][-1]
        average_intensity_seq = np.zeros(num_types[0])
        for event in seq['ci']:
            average_intensity_seq[int(event)] += 1

        average_intensity_seq /= T
        lambda_1 = lambda_1 + average_intensity_seq

    for seq in seq2[:num_seq]:
        T = seq['ti'][-1]
        average_intensity_seq = np.zeros(num_types[1])
        for event in seq['ci']:
            average_intensity_seq[int(event)] += 1

        average_intensity_seq /= T
        lambda_2 = lambda_2 + average_intensity_seq

    lambda_1 /= num_seq
    lambda_2 /= num_seq

    lambda_1 = lambda_1.unsqueeze(-1).numpy()
    lambda_2 = lambda_2.unsqueeze(-1).numpy()

    from sklearn.metrics.pairwise import euclidean_distances

    cost = euclidean_distances(lambda_1, lambda_2)
    print("lambda_1:", lambda_1)
    print("lambda_2:", lambda_2)
    print("cost", cost)
    np.savez("lambda1_lambda2_const_" + str(len(lambda_1)) + "_" + str(num_seq) + "_" + str(max_time) + ".npz",
             lambda_1=lambda_1, lambda_2=lambda_2, cost=cost)


if __name__ == "__main__":

    # path="exp_1000_1003_10000_0.3_time_1674901475.324254.pkl"
    # path = "exp_1135_1135_10000_0.3_time_1674904129.0491524.pkl"
    # #path = "exp_2708_2708_10000_0.12_time_1674904779.8678725.pkl"
    # max_time=0.3
    # num_seq = 8000
    # path=("./Tpp_data/Syn_data/unweighted/exp_100_100_2000_3_idx",3
    #       ,'lambda1_lambda2_const_100_2000_3_'
    #       ,"P_100_prior_0.0001_10000_2000_3_idx_")
    # path = ("./Tpp_data/Syn_data/unweighted/exp_10_10_2000_15_idx",15
    #         ,'lambda1_lambda2_const_10_2000_15_'
    #         ,"P_10_prior_0.0001_10000_2000_15_idx_")
    path = ("./Tpp_data/Syn_data/unweighted/exp/exp_50_50_2000_6_idx", 6, 'lambda1_lambda2_const_50_2000_6_',
            "./Tpp_data/Syn_data/unweighted/prior/P_50_prior_0.0001_10000_2000_6_idx_")

    # path = ("./Tpp_data/Real_data/exp_2708_2708_10000_2.3_idx", 2.3
    #         , 'lambda1_lambda2_const_2708_10000_2.3_'
    #         , "P_2708_prior_0.0001_10000_10000_2.3_idx_")
    for idx in range(5):
        plk_path = path[0]+str(idx)+".pkl"
        max_time = path[1]
        get_tpp_statistics(path=plk_path, num_seq=10000,
                           max_time=max_time, index=idx)

        abpath = path[2]+"idx"+str(idx)+".npz"

        get_prior_P(path0=plk_path, path1=abpath, epsilon=0.0001,
                    numItermax=10000, max_time=max_time, num_seq=10000, index=idx)

        Ppath = path[3]+str(idx)+".npz"
        prior_accuracy(path0=plk_path, path1=Ppath)
