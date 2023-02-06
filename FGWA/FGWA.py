import torch
import numpy as np
from fgwa_loss import distance_tensor
import torch.nn as nn
import matplotlib.pyplot as plt

class FGWA():

    def __init__(self, num_type_list, adj_list=None, device=None):
        super(FGWA, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.eps = np.finfo(float).eps

        self.num_type_list = num_type_list

        if adj_list == None:
            self.u_s = torch.ones((num_type_list[0], 1), dtype=torch.float32, device=self.device) / num_type_list[0]['num_type']
            self.u_t = torch.ones((num_type_list[1], 1), dtype=torch.float32, device=self.device) / num_type_list[1]['num_type']
        else:
            deg_s = torch.sum(adj_list[0], dim=1, keepdim=True)
            deg_t = torch.sum(adj_list[1], dim=1, keepdim=True)
            self.u_s = (torch.where(deg_s > 0, deg_s, torch.tensor(self.eps, dtype=torch.float32)) / torch.sum(adj_list[0])).to(self.device)
            self.u_t = (torch.where(deg_t > 0, deg_t, torch.tensor(self.eps, dtype=torch.float32)) / torch.sum(adj_list[1])).to(self.device)
        self.trans = torch.matmul(self.u_s, self.u_t.mT)

    def get_param(self):
        return self.model_list[0].mu, self.model_list[1].mu, self.model_list[0].infect, self.model_list[1].infect

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
            # if t <= 2:
            # # if torch.sum(torch.isnan(cost)).item() == 0:
            #     print('param: ', tau, self.trans)
            #     print('out_iter {}: {}'.format(t, cost))
            #     print(kernel)
            #     print('a, b: ', a[:, 0], b[:, 0])
            self.trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
            # if t % 100 == 0:
            #    print('sinkhorn iter {}/{}'.format(t, outer_iteration))
        cost = cost_mu_infect - 2 * alpha * torch.matmul(torch.matmul(infect_s, self.trans), infect_t.mT) + tau
        d_gw = (cost * self.trans).sum()
        return self.trans, d_gw, cost

    def gromov_wasserstein_distance(self, alpha):
        mu_s, mu_t, infect_s, infect_t = self.get_param()
        # L2 Distance
        cost_mu = distance_tensor(mu_s, mu_t)
        f1_infect = torch.sum(infect_s, dim=1, keepdim=True).repeat(1, infect_t.shape[0])
        f2_infect = torch.sum(infect_t, dim=0, keepdim=True).repeat(infect_s.shape[0], 1)
        cost_infect = f1_infect + f2_infect - 2 * torch.matmul(torch.matmul(infect_s, self.trans), infect_t.mT)
        cost = (1 - alpha) * cost_mu + alpha * cost_infect
        d_gw = (cost * self.trans).sum()
        return d_gw

    def plot_save(self, output_dir='.'):
        for idx, model in enumerate(self.model_list):
            model.plot_save(output_name='{}/Hawkes_param_{}.png'.format(output_dir, idx))
        trans = self.trans.data.cpu().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(trans)
        plt.colorbar()
        plt.savefig('Matching T.png')
        plt.close('all')









class HawkesProcess():

    def __init__(self, event_type, kernel, device=None):
        super(HawkesProcess, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.eps = np.finfo(float).eps
        self.max = np.finfo(float).max

        self.event_type = event_type
        self.mu = nn.Parameter(torch.rand(self.event_type, dtype=torch.float32, device=self.device))
        self.infect = nn.Parameter(
            torch.rand((self.event_type, self.event_type), dtype=torch.float32, device=self.device))
        self.kernel = kernel

        return

    def start(self, batch_size):
        self.nonnegative_clip()
        self.lambda_all = self.mu.clone().repeat((batch_size, 1))  # .clone() is important! Avoid inplace operation
        return

    def update(self, k, dt, mask):
        """
        k : event type
        dt : elapsed time since last event
        (batch_size, 1)
        """
        idx = mask.squeeze(1)
        k = k[idx]
        dt = dt[idx]

        alpha = self.infect[:, k.squeeze(1)].mT
        recur = self.kernel.values(dt) * (self.lambda_all[idx] - self.mu)
        self.lambda_all[idx] = alpha + recur + self.mu
        return

    def forward(self, k, dt, mask):
        self.update(k, dt, mask)
        return

    def compute_intensities(self, k, dt, mask):
        idx = mask.squeeze(1)
        k = k[idx]
        dt = dt[idx]
        recur = self.kernel.values(dt) * (self.lambda_all[idx].gather(1, k) - self.mu[k])
        lambda_value = self.mu[k] + recur
        return lambda_value

    def compute_total_intensity(self, dt, mask):
        idx = mask.squeeze(1)
        dt = dt[idx]
        recur = self.kernel.values(dt) * (self.lambda_all[idx] - self.mu)
        lambda_sum = self.mu + recur
        return lambda_sum

    def nonnegative_clip(self, threshold=None):
        if threshold == None:
            threshold = self.eps
        for param in self.parameters():
            w = param.data
            w[w < threshold] = threshold

    def update_param(self, mu, infect):
        self.mu.data.copy_(mu)
        self.infect.data.copy_(infect)
        return

