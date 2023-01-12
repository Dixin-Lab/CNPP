'''
self-attentive Hawkes process
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy

from Joint_sahp.models.embedding.event_type import TypeEmbedding
from Joint_sahp.models.embedding.position import PositionalEmbedding,BiasedPositionalEmbedding
from Joint_sahp.models.embedding.event_embedding import EventEmbedding
from Joint_sahp.models.attention.multi_head import MultiHeadedAttention
from Joint_sahp.models.utils.sublayer import SublayerConnection
from Joint_sahp.models.utils.feed_forward import PositionwiseFeedForward
from Joint_sahp.models.base import SeqGenerator, predict_from_hidden
from Joint_sahp.models.utils.gelu import GELU

from matplotlib import pyplot as plt
def map_index(seq,dic,base):
    #print("sahp,dic",dic)
    l_seq=seq.reshape(-1)
    l=len(l_seq)
    res_seq=torch.zeros_like(l_seq)
    for i in range(l):
        if l_seq[i].item() in dic.keys():
            res_seq[i]=dic[l_seq[i].item()]
        else:
            res_seq[i]=base
            #print("base")
    return res_seq.reshape(seq.shape).int()

class SAHP(nn.Module):
    "Generic N layer attentive Hawkes with masking"

    def __init__(self, nLayers, d_model, atten_heads, dropout, process_dim, device, max_sequence_length,tau=0.1,n_sink_iter=20):
        super(SAHP, self).__init__()


        self.process_num = len(process_dim)
        self.nLayers = nLayers
        self.process_dim = process_dim
        self.input_size = [ x + 1 for x in process_dim]
        print("self.input_size  ",self.input_size)
        self.query_size = d_model // atten_heads
        self.device = device
        self.gelu = GELU()

        self.d_model = d_model
        self.type_emb_list = nn.ModuleList([])
        print("self.process_dim[i]   ",self.process_dim)
        for i in range(self.process_num):#2共用
            self.type_emb_list.append(
                TypeEmbedding(self.input_size[i], d_model, padding_idx=self.process_dim[i]))

        self.shared_embedding = TypeEmbedding(self.input_size[0], d_model, padding_idx=self.process_dim[0])


        self.P = None


        self.f = nn.Sequential(
            nn.Linear(self.d_model,self.d_model, bias=True),
             nn.ReLU(),
            nn.Linear(self.d_model, self.input_size[1], bias=True),
            nn.ReLU()
        )
        self.X1=nn.Parameter(torch.FloatTensor(self.input_size[0],d_model))
        print("X1 shape  ",self.X1.shape)
        nn.init.normal_(self.X1)
        self.tau=tau
        self.n_sink_iter=n_sink_iter



        self.position_emb = BiasedPositionalEmbedding(d_model=d_model,max_len = max_sequence_length)
        self.attention = MultiHeadedAttention(h=atten_heads, d_model=self.d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=self.d_model, d_ff=self.d_model * 4, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=self.d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self.start_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True)
            ,nn.Softplus(beta=10.0)
        )

        self.intensity_layer_list = nn.ModuleList([])
        for i in range(self.process_num):
            self.intensity_layer_list.append(nn.Sequential(
                nn.Linear(self.d_model, self.process_dim[i], bias = True)
                ,nn.Softplus(beta=1.)
            ))  # 最后还有一处self.intensity_layer没改记得检查

    def state_decay(self, converge_point, start_point, omega, duration_t):
        # * element-wise product
        cell_t = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega * duration_t))
        return cell_t #softplus??

    def get_P(self,):
        from Joint_sahp.models.utils.sinkhorn import gumbel_sinkhorn
        P_before_GS=self.f(self.X1).T
        mask=torch.ones((self.input_size[1],self.input_size[0])).to(self.device)
        mask[:,-1]=0
        mask[-1,:]=0
        P_before_GS=P_before_GS.masked_fill(mask == 0, -1e9)

        return gumbel_sinkhorn(P_before_GS, tau=self.tau, n_iter=self.n_sink_iter)

    def forward(self, process_idx, seq_dt, seq_types, src_mask):
        if process_idx==0:
            type_embedding = self.X1[seq_types] * math.sqrt(self.d_model)
        else:
            #self.P=self.get_P()
            X2=self.get_P()@(self.X1.detach())
            type_embedding = X2[seq_types] * math.sqrt(self.d_model)
        #type_embedding=self.type_emb_list[process_idx](seq_types) * math.sqrt(self.d_model)
        # from Joint_sahp.models.utils.sinkhorn import gumbel_sinkhorn
        # self.P=gumbel_sinkhorn(self.type_emb_list[0].weight[:-1]@self.type_emb_list[0].weight[:-1].T, tau=self.tau, n_iter=self.n_sink_iter)

        position_embedding = self.position_emb(seq_types,seq_dt)

        x = type_embedding + position_embedding
        for i in range(self.nLayers):
            x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=src_mask))
            x = self.dropout(self.output_sublayer(x, self.feed_forward))

        embed_info = x



        self.start_point = self.start_layer(embed_info)
        self.converge_point = self.converge_layer(embed_info)
        self.omega = self.decay_layer(embed_info)

    def compute_loss(self, process_idx,seq_times, seq_onehot_types,n_mc_samples = 20):
        """
        Compute the negative log-likelihood as a loss function.

        Args:
            seq_times: event occurrence timestamps
            seq_onehot_types: types of events in the sequence, one hot encoded
            batch_sizes: batch sizes for each event sequence tensor, by length
            tmax: temporal horizon

        Returns:
            log-likelihood of the event times under the learned parameters

        Shape:
            one-element tensor
        """

        dt_seq = seq_times[:, 1:] - seq_times[:, :-1]
        cell_t = self.state_decay(self.converge_point, self.start_point, self.omega, dt_seq[:, :, None])
        intensity_layer = self.intensity_layer_list[process_idx]


        n_batch = seq_times.size(0)
        n_times = seq_times.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs = intensity_layer(cell_t)
        intens_at_evs = nn.utils.rnn.pad_sequence(
            intens_at_evs, padding_value=1.0,batch_first=True)  # pad with 0 to get rid of the non-events, log1=0
        log_intensities = intens_at_evs.log()  # log intensities
        seq_mask = seq_onehot_types[:, 1:]
        log_sum = (log_intensities * seq_mask).sum(dim=(2, 1))  # shape batch


        taus = torch.rand(n_batch, n_times, 1, n_mc_samples).to(device)# self.process_dim replaced 1
        taus = dt_seq[:, :, None, None] * taus  # inter-event times samples)

        cell_tau = self.state_decay(
            self.converge_point[:,:,:,None],
            self.start_point[:,:,:,None],
            self.omega[:,:,:,None],
            taus)
        cell_tau = cell_tau.transpose(2, 3)
        intens_at_samples = intensity_layer(cell_tau).transpose(2,3)
        intens_at_samples = nn.utils.rnn.pad_sequence(
            intens_at_samples, padding_value=0.0, batch_first=True)

        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC
        partial_integrals = dt_seq * total_intens_samples.mean(dim=2)

        integral_ = partial_integrals.sum(dim=1)

        res = torch.sum(- log_sum + integral_)
        return res


    def read_predict(self, process_idx, seq_times, seq_types, seq_lengths, pad, device,
                     hmax = 40, n_samples=1000, plot = False, print_info = False):
        """
        Read an event sequence and predict the next event time and type.

        Args:
            seq_times: # start from 0
            seq_types:
            seq_lengths:
            hmax:
            plot:
            print_info:

        Returns:

        """

        length = seq_lengths.item()  # exclude the first added event

        ## remove the first added event
        dt_seq = seq_times[1:] - seq_times[:-1]
        last_t = seq_times[length - 1]
        next_t = seq_times[length]

        dt_seq_valid = dt_seq[:length]  # exclude the last timestamp
        dt_seq_used = dt_seq_valid[:length-1]  # exclude the last timestamp
        next_dt = dt_seq_valid[length - 1]

        seq_types_valid = seq_types[1:length + 1]  # include the first added event
        from Joint_sahp.train_functions.train_sahp import MaskBatch
        last_type = seq_types[length-1]
        next_type = seq_types[length]
        if next_type == self.process_dim:
            print('Error: wrong next event type')
        seq_types_used = seq_types_valid[:-1]
        seq_types_valid_masked = MaskBatch(seq_types_used[None, :], pad, device)
        seq_types_used_mask = seq_types_valid_masked.src_mask


        with torch.no_grad():
            self.forward(process_idx, dt_seq_used, seq_types_used, seq_types_used_mask)

            if self.omega.shape[1] == 0:  # only one element
                estimate_dt, next_dt, error_dt, next_type, estimate_type = 0,0,0,0,0
                return estimate_dt, next_dt, error_dt, next_type, estimate_type

            elif self.omega.shape[1] == 1: # only one element
                converge_point = torch.squeeze(self.converge_point)[None, :]
                start_point = torch.squeeze(self.start_point)[None,:]
                omega = torch.squeeze(self.omega)[None, :]
            else:
                converge_point = torch.squeeze(self.converge_point)[-1, :]
                start_point = torch.squeeze(self.start_point)[-1, :]
                omega = torch.squeeze(self.omega)[-1, :]

            dt_vals = torch.linspace(0, hmax, n_samples + 1).to(device)
            h_t_vals = self.state_decay(converge_point,
                                        start_point,
                                        omega,
                                        dt_vals[:, None])
            if print_info:
                print("last event: time {:.3f} type {:.3f}"
                      .format(last_t.item(), last_type.item()))
                print("next event: time {:.3f} type {:.3f}, in {:.3f}"
                      .format(next_t.item(), next_type.item(), next_dt.item()))

            return predict_from_hidden(self, process_idx, h_t_vals, dt_vals, next_dt, next_type,
                                            plot, hmax, n_samples, print_info)


    def plot_estimated_intensity(self,timestamps, n_points=10000, plot_nodes=None,
                                 t_min=None, t_max=None,
                                 intensity_track_step=None, max_jumps=None,
                                 show=True, ax=None, qqplot=None):
        from simulation.simulate_hawkes import fuse_node_times
        event_timestamps, event_types = fuse_node_times(timestamps)

        event_timestamps = torch.from_numpy(event_timestamps)
        seq_times = torch.cat((torch.zeros_like(event_timestamps[:1]), event_timestamps),
                              dim=0).float()  # add 0 to the sequence beginning
        dt_seq = seq_times[1:] - seq_times[:-1]

        seq_types = torch.from_numpy(event_types)
        seq_types = seq_types.long()# convert from floattensor to longtensor

        intens_at_evs_lst = []
        sample_times = np.linspace(t_min, t_max, n_points)
        for i in range(self.process_dim):
            intens_at_samples, intens_at_evs = self.intensity_per_type(seq_types, dt_seq, sample_times, timestamps[i], type=i)
            intens_at_evs_lst.append(intens_at_samples)
            if qqplot is None:
                self._plot_tick_intensity(timestamps[i], sample_times, intens_at_samples,intens_at_evs,
                                          ax[i], i, n_points)
        if qqplot is not None:
            return intens_at_evs_lst

    def intensity_per_type(self, seq_types, dt_seq, sample_times, timestamps, type):
        from Joint_sahp.train_functions.train_sahp import MaskBatch

        intens_at_samples = []
        with torch.no_grad():

            onetype_length = timestamps.size
            alltype_length = len(seq_types)

            type_idx = np.arange(alltype_length)[seq_types == type]

            event_types_masked = MaskBatch(seq_types[None, :], pad=self.process_dim, device='cpu')
            event_types_mask = event_types_masked.src_mask

            self.forward(dt_seq, seq_types, event_types_mask)
            converge_point = torch.squeeze(self.converge_point)
            start_point = torch.squeeze(self.start_point)
            omega = torch.squeeze(self.omega)

            cell_t = self.state_decay(converge_point,
                                      start_point,
                                      omega,
                                      dt_seq[:, None])#

            intens_at_evs = torch.squeeze(self.intensity_layer(cell_t)).numpy()
            intens_at_evs = intens_at_evs[type_idx, type]


            event_idx = -1
            for t_time in sample_times:
                if t_time < timestamps[0]:
                    intens_at_samples.append(0)#np.zeros(self.process_dim)
                    continue

                if event_idx < onetype_length - 1 and t_time >= timestamps[event_idx + 1]:
                    event_idx += 1
                    # print(omega)

                aaa=dt_seq[:event_idx+1]
                bbb=seq_types[:event_idx+1]

                event_types_masked = MaskBatch(bbb[None, :], pad=self.process_dim, device='cpu')
                event_types_mask = event_types_masked.src_mask

                self.forward(aaa, bbb, event_types_mask)

                converge_point = torch.squeeze(self.converge_point)
                start_point = torch.squeeze(self.start_point)
                omega = torch.squeeze(self.omega)

                if omega.ndim == 2:
                    omega = omega[-1,:]
                    converge_point = converge_point [-1,:]
                    start_point = start_point[-1,:]
                cell_t = self.state_decay(converge_point,
                                          start_point,
                                          omega,
                                          t_time - timestamps[event_idx])#

                xxx = self.intensity_layer(cell_t).numpy()
                intens_at_samples.append(xxx[type])


            return intens_at_samples, intens_at_evs

    def _plot_tick_intensity(self, timestamps_i, sample_times, intensity_i, intens_at_evs,
                             ax, label, n_points):#
        x_intensity = np.linspace(sample_times.min(), sample_times.max(), n_points)
        y_intensity = intensity_i
        ax.plot(x_intensity, y_intensity)

        ax.set_title(label)


class SAHPGen(SeqGenerator):
    # sequence generator for the SAHP model

    def __init__(self,model, record_intensity = True):
        super(SAHPGen, self).__init__(model, record_intensity)
        self.lbda_ub = []

    def _restart_sequence(self):
        super(SAHPGen, self)._restart_sequence()
        self.lbda_ub = []
