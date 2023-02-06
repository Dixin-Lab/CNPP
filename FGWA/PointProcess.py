import torch
from loss import MaxLogLike, fill_triu
import torch.nn.functional as F


class PointProcessModel(torch.nn.Module):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """
    def __init__(self, num_type: int):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        """
        super().__init__()
        self.model_name = 'A Poisson Process'
        self.num_type = num_type
        self.mu = torch.nn.Parameter(torch.randn(self.num_type) * 0.5 - 2.0)
        self.loss_function = MaxLogLike()
        #self.fgwa=FGWA()

class MultiVariateHawkesProcessModel(PointProcessModel):
    """
        The class of generalized Hawkes process model
        contains most of necessary function.
        """

    def __init__(self, num_type: int, num_decay : int):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        :param num_decay: int, the number of decay functions
        """
        super().__init__(num_type)
        self.model_name = 'A MultiVariate Hawkes Process'
        self.num_decay = num_decay
        self.alpha = torch.nn.Parameter(torch.randn(self.num_type, self.num_type, self.num_decay) * 0.5 - 3.0)
        self.beta = torch.nn.Parameter(torch.randn(self.num_decay) * 0.5)

    def forward(self, event_times, event_types, input_mask, t0, t1):
        """
        :param event_times:  B x N
        :param input_mask:  B x  N
        :param t0: starting time
        :param t1: ending time
        :return: loglikelihood
        """

        mhat = F.softplus(self.mu)
        Ahat = F.softplus(self.alpha)
        omega  = F.softplus(self.beta)

        B, N = event_times.shape
        dim = mhat.shape[0]

        # Xu
        event_types -= 1
        event_types*=input_mask.long()

        # compute m_{u_i}
        mu = mhat[event_types]  # B x N


        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        dt = event_times[:, :, None] - event_times[:, None]  # (N, T, T)
        # print("event_times[:, None].shape",event_times[:, None].shape)
        # print("event_times[:, :, None].shape",event_times[:, :, None].shape)
        # print(dt.shape)
        # print("mu",mu)
        # print("omega",omega)
        #print("Amt",Ahat )
        dt = fill_triu(dt, 0)

        # kern[i,j] = omega* torch.exp(-omega*dt[i,j])
        kern = omega * torch.exp(-omega * dt)

        colidx = event_types.unsqueeze(1).repeat(1, N, 1)
        rowidx = event_types.unsqueeze(2).repeat(1, 1, N)

        Auu = Ahat[rowidx, colidx].squeeze(dim=3)

        ag = Auu * kern
        ag = fill_triu(ag, 0)
        # print('ag',ag.shape)
        # compute total rates of u_i at time i
        rates = mu + torch.sum(ag, dim=2)

        #baseline \sum_i^dim \int_0^T \mu_i
        compensator_baseline = (t1 - t0) * torch.sum(mhat)

        # \int_{t_i}^T \omega \exp{ -\omega (t - t_i )  }
        log_kernel = -omega * (t1[:, None] - event_times)
        Int_kernel = (1 - torch.exp(log_kernel)).unsqueeze(1)

        Au = Ahat[:, event_types].permute(1, 0, 2, 3).squeeze(3)

        Au_Int_kernel = (Au * Int_kernel).sum(dim=1) * input_mask

        compensator = compensator_baseline + Au_Int_kernel.sum(dim=1)


        loglik = torch.log(rates + 1e-8).mul(input_mask).sum(-1)  #

        # print("rates", rates)
        # print("loglik", loglik)
        return (loglik, compensator)



class HawkesPointProcess(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mu = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5 - 2.0)
        self.alpha = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5 - 3.0)
        self.beta = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5)

    def logprob(self, event_times, input_mask, t0, t1):
        """
        :param event_times:
        :param input_mask:
        :param t0:
        :param t1:
        :return:
        """

        mu = F.softplus(self.mu)
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)

        dt = event_times[:, :, None] - event_times[:, None]  # (N, T, T)
        dt = fill_triu(-dt * beta, -1e20)
        lamb = torch.exp(torch.logsumexp(dt, dim=-1)) * alpha * beta + mu  # (N, T)
        loglik = torch.log(lamb + 1e-8).mul(input_mask).sum(-1)  # (N,)

        log_kernel = -beta * (t1[:, None] - event_times) * input_mask + (1.0 - input_mask) * -1e20

        compensator = (t1 - t0) * mu
        compensator = compensator - alpha  * (torch.exp(torch.logsumexp(log_kernel, dim=-1)) - input_mask.sum(-1))

        return (loglik- compensator)






