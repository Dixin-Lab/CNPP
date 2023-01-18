import numpy as np
import pickle
element_max = np.vectorize(max)
import torch
import matplotlib.pyplot as plt
def sinkhorn(C, a, b, epsilon, precision):
    a = a.reshape((C.shape[0], 1))
    b = b.reshape((C.shape[1], 1))
    K = np.exp(-C / epsilon)

    # initialization
    u = np.ones((C.shape[0], 1))
    v = np.ones((C.shape[1], 1))
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    p_norm = np.trace(P.T @ P)

    while True:
        u = a / element_max((K @ v), 1e-300)  # avoid divided by zero
        v = b / element_max((K.T @ u), 1e-300)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        if abs((np.trace(P.T @ P) - p_norm) / p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    return P, np.trace(C.T @ P)


def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        num_types = [data['pmat'].shape[0],data['pmat'].shape[1]]

        return data, num_types

if __name__=="__main__":
    data,num_types=load_data("exp_10_10_2000.pkl")
    train_seqs=data['train']
    average_intensity=torch.zeros(sum(num_types))
    #print(train_seqs[0])
    print(len(train_seqs))
    for seq in train_seqs:
        T= seq['ti'][-1]
        average_intensity_seq=np.zeros(sum(num_types))
        for event in seq['ci']:
            average_intensity_seq[event]+=1

        average_intensity_seq/=T
        #print(average_intensity_seq)
        average_intensity=average_intensity+average_intensity_seq

    average_intensity/=len(train_seqs)

    a=average_intensity[:num_types[0]].unsqueeze(-1).numpy()
    b = average_intensity[num_types[0]:].unsqueeze(-1).numpy()
    # a=a/sum(a)
    # b=b/sum(b)
    from sklearn.metrics.pairwise import euclidean_distances
    cost = euclidean_distances(a,b)
    print("a:",a)
    print("b:",b)
    print("cost",cost)
    epsilon=0.00001
    distribute_a=np.ones(num_types[0])/num_types[0]
    distribute_b = np.ones(num_types[1]) / num_types[1]
    P,wd=sinkhorn(cost, distribute_a,distribute_b, epsilon=epsilon, precision=1e-30)
    np.savez("P_prior"+str(epsilon)+".npz",P=P)
    print("P",sum(sum(P)))
    plt.figure(figsize=(10, 5))
    plt.imshow(P)
    plt.colorbar()
    plt.savefig('./Plot/P'+str(epsilon)+'.pdf')
    plt.show()

