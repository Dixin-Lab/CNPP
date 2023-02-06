import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import torch

def acc_score(emb1,emb2,gnd,topk=5,type="cosin"):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_dis = cosine_similarity(emb1.cpu().detach().numpy(),emb2.cpu().detach().numpy())
    print("HitK",topk,"score:",topk_alignment_score(cosine_dis,gnd,topk,right=1))

def acc_score_P(P,gnd,topk=5):
    dis=P
    print("HitK",topk,"score:",topk_alignment_score(dis,gnd,topk,right=1))



def matching(alpha):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)




def topk_alignment_score(sim,gnd,topk,right=1):
    """
    :param sim: n1xn2 similarity matrix (ndarray)
    :param gnd: numx2 the gnd pairs, and the label is beginning with 0.
    :param topk:
    :return: accuracy scores
    """
    possible_alignment = np.argsort(sim, axis=1)
    num = 0
    length=gnd.shape[0]
    #print("gnd",gnd)
    for idx in range(length):
        if gnd[idx, right] in possible_alignment[gnd[idx, 1-right]][-topk:]:
            num += 1
    return num / length