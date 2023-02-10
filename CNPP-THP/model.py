import math
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


class CoupledEmbedding(nn.Module):
    def __init__(
            self,
            num_types: List, dim: int, n_iter: int = 20, P_prior=None,is_param=False):
        super().__init__()
        self.total_num = sum(num_types) + 1
        self.num_types = num_types
        self.n_iter = n_iter
        self.src_emb = nn.Linear(num_types[0] + 1, dim, bias=False)
        self.P=P_prior
        self.is_param = is_param
        self.P_=nn.Parameter(P_prior*min(self.num_types[1],self.num_types[0]))
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


    def forward(self, process_idx, event_types: torch.Tensor,is_param=False):
        """
        :param event_types: [batch size, seq length]
        :return:
            [batch size, seq length, dim]
        """
        event_types_onehot = F.one_hot(event_types, num_classes=self.num_types[process_idx]+1)  # [batch size, seq length, total_num]
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

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb_list = nn.ModuleList([nn.Embedding(event_type + 1, d_model, padding_idx=PAD)\
                             for event_type in num_types])

        self.event_emb = CoupledEmbedding(num_types, d_model, n_iter=100, P_prior=P_prior, is_param=is_param)


        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, process_idx, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # event_type bxl
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        #coupling
        #enc_output = self.event_emb_list[process_idx](event_type)

        enc_output = self.event_emb(process_idx,event_type)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


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


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types: list, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, P_prior=None, is_param=False):
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

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear_list = nn.ModuleList([nn.Linear(d_model, event_type) for event_type in num_types])

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

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
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor_list[process_idx](enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)