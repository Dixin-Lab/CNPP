import numpy as np
import torch
import  matplotlib.pyplot as plt
import torch.nn.functional as F
from PointProcess import MultiVariateHawkesProcessModel
from fgwa_loss import distance_tensor
class HPwithFGWA(torch.nn.Module):
    def __init__(self, num_type_list, adj_list=None, device=None):
        super(HPwithFGWA, self).__init__()
        self.device = torch.device(device)

        self.num_type_list=num_type_list
        self.hp0=MultiVariateHawkesProcessModel(num_type_list[0],1).to(self.device)
        self.hp1=MultiVariateHawkesProcessModel(num_type_list[1],1).to(self.device)


        self.eps = np.finfo(float).eps

        if adj_list == None:
            self.u_s = torch.ones((num_type_list[0], 1), dtype=torch.float32, device=self.device) / num_type_list[0]
            self.u_t = torch.ones((num_type_list[1], 1), dtype=torch.float32, device=self.device) / num_type_list[1]
        else:
            deg_s = torch.sum(adj_list[0], dim=1, keepdim=True)
            deg_t = torch.sum(adj_list[1], dim=1, keepdim=True)
            self.u_s = (torch.where(deg_s > 0, deg_s, torch.tensor(self.eps, dtype=torch.float32)) / torch.sum(
                adj_list[0])).to(self.device)
            self.u_t = (torch.where(deg_t > 0, deg_t, torch.tensor(self.eps, dtype=torch.float32)) / torch.sum(
                adj_list[1])).to(self.device)
        self.trans = torch.matmul(self.u_s, self.u_t.mT)


    def get_param(self):
        return F.softplus(self.hp0.mu), F.softplus(self.hp1.mu), F.softplus(self.hp0.alpha.squeeze()), F.softplus(self.hp1.alpha.squeeze())

    def gromov_wasserstein_learning(self, outer_iteration, inner_interation, alpha, tau):
        mu_s, mu_t, infect_s, infect_t = self.get_param()
        mu_s, mu_t, infect_s, infect_t = mu_s.detach(), mu_t.detach(), infect_s.detach(), infect_t.detach()

        a = self.u_s
        b = 0

        # L2 Distance
        cost_mu = (mu_s.unsqueeze(-1) - mu_t.unsqueeze(0)) ** 2
        f1_infect = torch.matmul(infect_s ** 2, self.u_s).repeat(1, infect_t.shape[0])
        f2_infect = torch.matmul(self.u_t.mT, infect_t ** 2).repeat(infect_s.shape[0], 1)
        cost_mu_infect = (1 - alpha) * cost_mu + alpha * (f1_infect + f2_infect)
        for t in range(outer_iteration):
            cost = cost_mu_infect - 2 * alpha * torch.matmul(torch.matmul(infect_s, self.trans), infect_t.mT) + tau
            kernel = torch.exp(-cost / tau) * self.trans
            for l in range(inner_interation):
                # if t <= 2:
                #     print(torch.matmul(kernel.mT, a).squeeze())
                b = self.u_t / torch.matmul(kernel.mT, a)
                a = self.u_s / torch.matmul(kernel, b)

            self.trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))

        return

    def gromov_wasserstein_distance(self, alpha):
        mu_s, mu_t, infect_s, infect_t = self.get_param()
        # L2 Distance
        cost_mu = distance_tensor(mu_s, mu_t)
        f1_infect = torch.sum(infect_s, dim=1, keepdim=True).repeat(1, infect_t.shape[0])
        f2_infect = torch.sum(infect_t, dim=0, keepdim=True).repeat(infect_s.shape[0], 1)
        cost_infect = f1_infect + f2_infect - 2 * torch.matmul(torch.matmul(infect_s, self.trans), infect_t.mT)
        cost = (1 - alpha) * cost_mu + alpha * cost_infect
        #detach by lqb
        d_gw = (cost * self.trans.detach()).sum()

        return d_gw



