import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from constant import PAD
from module import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2, bias=False)

    def forward(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        duration_t = F.pad(time[:, 1:] - time[:, :-1], (1, 0), 'constant', 0)

        phi = self.Wt(duration_t.unsqueeze(-1))
        aa = len(time.size())
        if aa > 1:
            length = time.size(1)
        else:
            length = time.size(0)
        # print("phi size",phi.size())

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        #         print("result size",result.size())
        #         print("result[:, :, 0::2] size",result[:, :, 0::2].size())

        #         result[:, :, 0::2] = torch.sin(result[:, :, 0::2] + phi)
        #         result[:, :, 1::2] = torch.cos(result[:, :, 1::2] + phi)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        result = torch.cat([pe_sin, pe_cos], dim=-1)

        return result * non_pad_mask


class CoupledEmbedding(nn.Module):
    def __init__(
            self,
            num_types: List, dim: int, n_iter: int = 20, P_prior=None, is_param=False):
        super().__init__()
        self.total_num = sum(num_types) + 1
        self.num_types = num_types
        self.n_iter = n_iter
        self.src_emb = nn.Linear(num_types[0] + 1, dim, bias=False)
        self.P = P_prior
        self.is_param=is_param
        self.P_ = nn.Parameter(P_prior * min(self.num_types[1], self.num_types[0]))
        # How to initialize coupling might be important. I am not sure.
        self.coupling = nn.Parameter(data=torch.randn(num_types[1], num_types[0]))
        self.f = nn.Sequential(
            nn.Linear(dim, num_types[1], bias=True),
            nn.ReLU()
        )

    def sinkhorn(self):
        tau=0.1
        if self.is_param:
            X1=torch.eye(self.num_types[0] + 1).to('cuda')
            log_alpha = -self.f(self.src_emb(X1)[1:]).T/tau
        if not self.is_param:
            log_alpha = -self.coupling/ tau

        for _ in range(self.n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)

        return log_alpha.exp() #* self.num_types[1]


    def forward(self, process_idx, event_types: torch.Tensor):
        """
        :param event_types: [batch size, seq length]
        :return:
            [batch size, seq length, dim]
        """
        event_types_onehot = F.one_hot(event_types, num_classes=self.num_types[
                                                                    process_idx] + 1)  # [batch size, seq length, total_num]
        event_types_onehot = event_types_onehot.type(torch.FloatTensor)
        event_types_onehot = event_types_onehot.to(event_types.device)
        trans = self.sinkhorn()  # num1 x num0

        if process_idx == 0:
            event_types_aligned = event_types_onehot[:, :, :(self.num_types[0] + 1)].clone()
        elif process_idx == 1:
            event_types_aligned = event_types_onehot[:, :, :(self.num_types[0] + 1)].clone()
            event_types_aligned[:, :, 1:] = torch.matmul(event_types_onehot[:, :, 1:],
                                                         trans)
            #trans.detach())

        return self.src_emb(event_types_aligned)


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, P_prior, is_param):
        super().__init__()

        self.d_model = d_model

        self.temporal_enc = BiasedPositionalEmbedding(d_model)

        # coupling emb
        self.event_emb = CoupledEmbedding(num_types, d_model, n_iter=30, P_prior=P_prior,is_param=is_param)

        # event type embedding
        self.event_emb_list = nn.ModuleList([nn.Embedding(event_type + 1, d_model, padding_idx=PAD) \
                                             for event_type in num_types])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def forward(self, process_idx, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # event_type bxl
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        #enc_output = self.event_emb_list[process_idx](event_type)
        enc_output = self.event_emb(process_idx,event_type)
        ################################################################
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()
        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SAHP(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types: list, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, P_prior=None,is_param=False):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            P_prior=P_prior,
            is_param=is_param
        )
        self.gelu = GELU()
        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear_list = nn.ModuleList([nn.Linear(d_model, event_type) for event_type in num_types])

        self.start_layer = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True)
            , nn.Softplus(beta=10.0)
        )

        self.intensity_layer_list = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, event_type, bias=True)
            , nn.Softplus(beta=1.)
        ) for event_type in num_types])

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor_list = nn.ModuleList([Predictor(d_model, event_type) for event_type in num_types])

    def forward(self, process_idx, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(process_idx, event_type, event_time, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor_list[process_idx](enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

    def compute_event(self, event, non_pad_mask):
        """ Log-likelihood of events. """

        # add 1e-9 in case some events have 0 likelihood
        event += math.pow(10, -9)
        event.masked_fill_(~non_pad_mask.bool(), 1.0)

        result = torch.log(event)
        return result

    def compute_integral_unbiased(self, process_idx, data, time, non_pad_mask, type_mask):
        """ Log-likelihood of non-events, using Monte Carlo integration. """

        num_samples = 50

        diff_time = (torch.cat([time[:, 0].reshape(-1, 1), time[:, 1:] - time[:, :-1]], dim=1)) * non_pad_mask[:, :]

        taus = torch.rand(*diff_time.size(), 1, num_samples).to(time.device)  # self.process_dim replaced 1
        taus = diff_time[:, :, None, None] * taus  # inter-event times samples)

        start_point = self.start_layer(data)
        converge_point = self.converge_layer(data)
        omega = self.decay_layer(data)

        cell_tau = torch.tanh(converge_point[:, :, :, None]
                              + (start_point[:, :, :, None] - converge_point[:, :, :, None])
                              * torch.exp(- omega[:, :, :, None] * taus))

        cell_tau = cell_tau.transpose(2, 3)
        intens_at_samples = self.intensity_layer_list[process_idx](cell_tau).transpose(2, 3)

        total_intens_samples = intens_at_samples.sum(dim=2)  # shape batch * N * MC

        partial_integrals = diff_time * total_intens_samples.mean(dim=2)

        integral_ = partial_integrals.sum(dim=1)

        return integral_

    def log_likelihood(self, process_idx, data, time, types):
        non_pad_mask = get_non_pad_mask(types).squeeze(2)

        num_types = self.num_types[process_idx]

        type_mask = torch.zeros((*types.size(), num_types)).to(data.device)
        for i in range(num_types):
            type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
        #######################################################################
        diff_time = (torch.cat([time[:, 0].reshape(-1, 1), time[:, 1:] - time[:, :-1]], dim=1)) * non_pad_mask[:, :]
        start_point = self.start_layer(data)
        converge_point = self.converge_layer(data)
        omega = self.decay_layer(data)
        cell_t = torch.tanh(
            converge_point + (start_point - converge_point) * torch.exp(- omega * diff_time[:, :, None]))

        all_lambda = self.intensity_layer_list[process_idx](cell_t)

        # print("log_likelihood",all_lambda)
        type_lambda = torch.sum(all_lambda * type_mask, dim=2)

        # event log-likelihood

        event_ll = self.compute_event(type_lambda, non_pad_mask)
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, either numerical integration or MC integration
        # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
        non_event_ll = self.compute_integral_unbiased(process_idx, data, time, non_pad_mask, type_mask)
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll
