import torch
import torch.nn.functional as F


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
    

def sinkhorn2(cost: torch.Tensor, tau: float, num: int, eps: float = 1e-6):
    x = -cost / tau
    for _ in range(num):
        x = x - torch.logsumexp(x, dim=1, keepdim=True)
        x = x - torch.logsumexp(x, dim=0, keepdim=True)
    return torch.exp(x)


# 2022.12.13 train_pairs索引记得改一改
def cl_loss(emb_list, train_pairs, neg_samples, gamma):
    anchor0, anchor1 = train_pairs[:, 0].unsqueeze(1), train_pairs[:, 1].unsqueeze(1)
    emb0, emb1 = emb_list[0], emb_list[1]
    
    anchor0_emb, anchor1_emb = emb0[anchor0], emb1[anchor1]
    neg0, neg1 = get_neg_nodes(emb0, emb1, neg_samples, anchor0, anchor1)
    neg0_emb, neg1_emb = emb0[neg0], emb1[neg1]

    # L1_loss
    A = torch.sum(torch.abs(anchor0_emb - anchor1_emb), 1)
    D = A + gamma
    B1 = -torch.sum(torch.abs(anchor0_emb - neg0_emb), 1)
    L1 = torch.sum(F.relu(B1 + D))
    B2 = -torch.sum(torch.abs(anchor1_emb - neg1_emb), 1)
    L2 = torch.sum(F.relu(B2 + D))
    return (L1 + L2) / train_pairs.shape[0]

def get_neg_nodes(out0, out1, neg_nodes, anchor0, anchor1):
    anchor0_vec = out0[anchor0].squeeze(1)
    anchor1_vec = out1[anchor1].squeeze(1)
    sim0 = torch.cdist(anchor0_vec, out1, p=1)
    neg0 = sim0.argsort()[:, :neg_nodes]
    sim1 = torch.cdist(anchor1_vec, out0, p=1)
    neg1 = sim1.argsort()[:, :neg_nodes]
    return neg0, neg1