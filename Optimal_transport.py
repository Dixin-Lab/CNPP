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
        print("precision", abs((np.trace(P.T @ P) - p_norm) / p_norm))
        if abs((np.trace(P.T @ P) - p_norm) / p_norm) < precision:
            break
        p_norm = np.trace(P.T @ P)
    return P, np.trace(C.T @ P)


def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
        num_types = [data['pmat'].shape[0],data['pmat'].shape[1]]

        return data, num_types


def get_prior_P(path0, path1, epsilon=0.0001, precision=1e-11,max_time=0, num_seq=2000):

    data, num_types = load_data(path0)

    a_b_cost = np.load(path1)
    cost=a_b_cost['cost']
    a=a_b_cost['a']

    distribute_a = np.ones(num_types[0]) / num_types[0]
    distribute_b = np.ones(num_types[1]) / num_types[1]
    P, wd = sinkhorn(cost, distribute_a, distribute_b, epsilon=epsilon, precision=precision)
    np.savez("P_" + str(len(a)) + "_prior_" + str(epsilon) +"_"+str(precision)+"_"+str(num_seq)+"_"+str(max_time)+".npz", P=P)
    print("P", sum(sum(P)))
    plt.figure(figsize=(10, 5))
    plt.imshow(P)
    plt.colorbar()
    # plt.savefig('./Plot/P'+str(epsilon)+'.pdf')
    plt.show()
    plt.close()

    pmat = data['pmat']
    plt.figure(figsize=(10, 5))
    plt.imshow(pmat)
    plt.colorbar()
    plt.savefig('./Plot/P_'+str(epsilon)+'_'+str(len(a))+'.pdf')
    plt.show()
    plt.close()

def prior_accuracy(path0,path1):
    data, num_types = load_data(path0)
    pmat = data['pmat']
    P_prior= np.load(path1)['P']
    print(P_prior.shape)
    gnd_pair = np.argwhere(pmat.numpy()).astype(np.int32)
    from Joint_sahp.metric import acc_score_P
    for topk in [1, 3, 5]:  # ,30,50
        acc_score_P(P_prior, gnd=gnd_pair[:, ], topk=topk)
        # acc_score_P(pmt, gnd=gnd_pair[:, ], topk=topk)


if __name__=="__main__":
    # data,num_types=load_data("./exp_100_100_2000_20.pkl")
    # print(type(data['test']))
    # train_mid=len(data['train'])//2
    # test_mid=len(data['test'])//2
    # need_seq_num=1500
    # train_seqs=data['train']
    # average_intensity=torch.zeros(sum(num_types))
    # #print(train_seqs[0])
    # print(len(train_seqs))
    # for idx in range(need_seq_num):
    #     T= train_seqs[idx]['ti'][-1]
    #     average_intensity_seq=np.zeros(sum(num_types))
    #     #print(len(train_seqs[idx]['ci']))
    #     for event in train_seqs[idx]['ci']:
    #         #print("event ",type(event))
    #         average_intensity_seq[int(event)]+=1
    #     average_intensity_seq/=T
    #     average_intensity=average_intensity+average_intensity_seq
    #
    #     T = train_seqs[idx+train_mid]['ti'][-1]
    #     average_intensity_seq = np.zeros(sum(num_types))
    #     for event in train_seqs[idx+train_mid]['ci']:
    #         # print("event ",type(event))
    #         average_intensity_seq[int(event)] += 1
    #     average_intensity_seq /= T
    #     average_intensity = average_intensity + average_intensity_seq
    #
    #
    # average_intensity/=need_seq_num
    #
    # a = average_intensity[:num_types[0]].unsqueeze(-1).numpy()
    # b = average_intensity[num_types[0]:].unsqueeze(-1).numpy()
    # # a=a/sum(a)
    # # b=b/sum(b)
    # from sklearn.metrics.pairwise import euclidean_distances
    # cost = euclidean_distances(a,b)
    # print("a:",a)
    # print("b:",b)
    # print("cost",cost)
    # np.savez("a_b_const_" + str(len(a))+"_"+str(need_seq_num)+"_20" + ".npz", a=a,b=b,cost=cost)
    #
    #
    # data,num_types=load_data("./Joint_THP/exp_50_50_2000_45.pkl")
    # data['train'].extend(data['test'])
    # train_seqs = data['train']
    # average_intensity=torch.zeros(sum(num_types))
    # #print(train_seqs[0])
    # print(len(train_seqs))
    # for seq in train_seqs:
    #     T= seq['ti'][-1]
    #     average_intensity_seq=np.zeros(sum(num_types))
    #     for event in seq['ci']:
    #         #print("event ",type(event))
    #         average_intensity_seq[int(event)]+=1
    #
    #     average_intensity_seq/=T
    #     #print(average_intensity_seq)
    #     average_intensity=average_intensity+average_intensity_seq
    #
    # average_intensity/=len(train_seqs)/2
    #
    # a=average_intensity[:num_types[0]].unsqueeze(-1).numpy()
    # b = average_intensity[num_types[0]:].unsqueeze(-1).numpy()
    # # a=a/sum(a)
    # # b=b/sum(b)
    # from sklearn.metrics.pairwise import euclidean_distances
    # cost = euclidean_distances(a,b)
    # print("a:",a)
    # print("b:",b)
    # print("cost",cost)
    # np.savez("a_b_const_" + str(len(a))+"_"+str(2000)+"_45" + ".npz", a=a,b=b,cost=cost)



    num_seq=2000
    max_time=5
    get_prior_P('./Joint_THP/exp_50_50_2000_45.pkl','a_b_const_50_2000_45.npz',epsilon=0.0001,precision=1e-11,max_time=max_time,num_seq=num_seq)
    #prior_accuracy('./Joint_THP/exp_50_50_2000_45.pkl','P_50_prior_0.0001_1e-11_2000_45.npz')



    # for i in [5,10,15,20]:
    #     path="P_100_prior_0.0001_1e-11_"+str(i)+".npz"
    #     P=np.load(path)['P']
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(P)
    #     plt.colorbar()
    #     plt.show()
    #     plt.title(str(i))
    #     plt.close()