import numpy as np
import torch
import torch.utils.data
from torch.distributions import categorical, exponential, uniform
from typing import Dict, List, Tuple
from constant import PAD
import pickle

# simulator
def exp_kernel(dt: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    gt = w * torch.exp(-w * dt)
    gt[dt < 0] = 0
    return gt


def _conditional_intensity_vector(seq: Dict,
                                  t: torch.Tensor,
                                  infect: torch.Tensor,
                                  mu: torch.Tensor,
                                  w: float = 1.0) -> torch.Tensor:
    lambda_t = torch.clone(mu)
    if seq['ti'] is not None:
        dt = t - seq['ti']
        gt = exp_kernel(dt, w)
        lambda_t += torch.sum(infect[:, seq['ci']] * gt, dim=1)
    return lambda_t


def simulate_ogata_thinning(mu: torch.Tensor,
                            infect: torch.Tensor,
                            num_seq: int,
                            max_time: float = 5.0,
                            w: float = 1.0) -> Tuple[List, List]:
    seqs = []
    seqs_len = []
    for n in range(num_seq):
        sequence = {'ti': None,  # event_times
                    'ci': None,  # event_type
                    'time_since_last_event': None}

        t = 0.0
        lambda_t = torch.clone(mu)
        lambda_all = torch.sum(lambda_t)

        exp_dist = exponential.Exponential(rate=lambda_all)
        unif_dist = uniform.Uniform(low=0.0, high=1.0)

        duration = 0.0
        while t < max_time:
            s = exp_dist.sample()
            u = unif_dist.sample()
            lambda_ts = _conditional_intensity_vector(seq=sequence, t=t + s, infect=infect, mu=mu, w=w)
            lambda_ts_all = torch.sum(lambda_ts)
            lambda_normal = lambda_ts / lambda_ts_all

            t += s
            duration += s

            if t < max_time and u < lambda_ts_all / lambda_all:
                cat_dist = categorical.Categorical(probs=lambda_normal)
                c = cat_dist.sample()
                c = torch.LongTensor([c])
                t_tensor = torch.Tensor([t])
                duration_tensor = torch.Tensor([duration])
                if sequence['ti'] is None:
                    sequence['ti'] = t_tensor
                    sequence['ci'] = c
                    sequence['time_since_last_event'] = duration_tensor
                else:
                    sequence['ti'] = torch.cat([sequence['ti'], t_tensor], dim=0)
                    sequence['ci'] = torch.cat([sequence['ci'], c], dim=0)
                    sequence['time_since_last_event'] = torch.cat([sequence['time_since_last_event'], duration_tensor],
                                                                  dim=0)

                duration = torch.zeros(1)

            exp_dist = exponential.Exponential(rate=lambda_ts_all)

        if sequence['ci'] is not None:
            print(len(sequence['ci']))
            seqs_len.append(len(sequence['ci']))
            seqs.append(sequence)

    print('Seqs_len: ', seqs_len)
    print('Seqs: ', seqs)
    return seqs, seqs_len


def make_seq(params: Dict, num_seq: int, max_time: float, w: float = 1.0) -> Tuple[List, List]:
    sequences, seqs_len = simulate_ogata_thinning(params['mu'], params['A'], num_seq, max_time, w)
    return sequences, seqs_len


def synthetic_hawkes_parameters(dim: int = 10, thres: float = 0.5) -> Tuple[Dict, Dict, torch.Tensor]:
    mu1 = torch.rand(dim)
    infect1 = torch.rand(dim, dim)
    infect1[infect1 < thres] = 0
    infect1 = 0.8 * infect1 / torch.linalg.svdvals(infect1)[0]
    idx = torch.randperm(dim)
    mu2 = mu1[idx]
    infect2 = infect1[idx, :]
    infect2 = infect2[:, idx]
    pmat = torch.eye(dim)
    pmat = pmat[:, idx]
    params1 = {'mu': mu1, 'A': infect1}
    params2 = {'mu': mu2, 'A': infect2}
    return params1, params2, pmat


def generate_synthetic_tpps(dim: int = 10,
                            num_seq: int = 200,
                            max_time: float = 30,
                            thres: float = 0.5,
                            w: float = 0.1) -> Tuple[Dict, List, torch.Tensor]:
    params1, params2, pmat = synthetic_hawkes_parameters(dim=dim, thres=thres)
    seqs1, seqs_len1 = make_seq(params=params1, num_seq=num_seq, max_time=max_time, w=w)
    seqs2, seqs_len2 = make_seq(params=params2, num_seq=num_seq, max_time=max_time, w=w)
    for n in range(len(seqs2)):
        seqs2[n]['ci'] = seqs2[n]['ci'] + dim
    seqs_train = seqs1[:-100] + seqs2[:-100]
    seqs_len_train = seqs_len1[:-100] + seqs_len2[:-100]
    seqs_test = seqs1[-100:] + seqs2[-100:]
    seqs_len_test = seqs_len1[-100:] + seqs_len2[-100:]
    return {'train':seqs_train, 'test':seqs_test}, [params1, params2], pmat

    #return {'train': [seqs_train, seqs_len_train], 'test': [seqs_test, seqs_len_test]}, [params1, params2], pmat

def generate_synthetic_tpps_from_graph_data(params1,params2,dim: int = 10,
                            num_seq: int = 200,
                            max_time: float = 30,
                            thres: float = 0.5,
                            w: float = 0.1) -> Tuple[Dict, List]:
    # params1, params2= synthetic_hawkes_parameters(dim=dim, thres=thres)
    seqs1, seqs_len1 = make_seq(params=params1, num_seq=num_seq, max_time=max_time, w=w)
    seqs2, seqs_len2 = make_seq(params=params2, num_seq=num_seq, max_time=max_time, w=w)
    for n in range(len(seqs2)):
        seqs2[n]['ci'] = seqs2[n]['ci'] + dim
    seqs_train = seqs1[:-100] + seqs2[:-100]
    seqs_len_train = seqs_len1[:-100] + seqs_len2[:-100]
    seqs_test = seqs1[-100:] + seqs2[-100:]
    seqs_len_test = seqs_len1[-100:] + seqs_len2[-100:]
    return {'train':seqs_train, 'test':seqs_test}, [params1, params2]



def save_pkl(data_dict,params1,params2,pmat, max_time, seq_num, output_dir='.'):
    data = {}
    data['params1'] = params1
    data['params2'] = params2
    data['pmat'] = pmat
    data['train'] = data_dict['train']
    data['test'] = data_dict['test']
    num1=pmat.shape[0]
    num2=pmat.shape[1]


    with open('{}/exp_{}_{}_{}_{}.pkl'.format(output_dir,num1,num2,seq_num,max_time), 'wb') as f:
        pickle.dump(data, f)
    return



#tick

def make_seq_from_tick(infect, w, n_nodes, mu, max_time = 30, num_seq =2000):

    from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti

    infect= infect[:, :, np.newaxis]
    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=infect, decays=w, baseline=mu,
        end_time=max_time, verbose=False)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=num_seq)

    multi.end_time = [max_time for i in range(num_seq)]
    multi.simulate()


    seqs = []
    seqs_len = []
    timestamps = multi.timestamps
    for i in range(num_seq):

        sequence = {'ti': None,  # event_times
                    'ci': None,  # evnet_type
                    }
        ti_array = np.empty(0)
        ci_array = np.empty(0)
        for j in range(n_nodes):
            array = timestamps[i][j]
            l = len(array)
            ti_array = np.hstack((ti_array, array))
            ci_array = np.hstack((ci_array,np.full(l,j)))
            sorted_idx = np.argsort(ti_array)
            ti_array = ti_array[sorted_idx]
            ci_array = ci_array[sorted_idx]

        sequence['ti'] = ti_array
        sequence['ci'] = ci_array.astype(int)
        seqs.append(sequence)
        seqs_len.append(len(ci_array))
        print(i, len(ci_array))

    return seqs,seqs_len

def generate_synthetic_tpps_from_tick(dim: int = 10,
                            num_seq: int = 200,
                            max_time: float = 30,
                            thres: float = 0.5,
                            w: float = 0.1) -> Tuple[Dict, List, torch.Tensor]:
    params1, params2, pmat = synthetic_hawkes_parameters(dim=dim, thres=thres)
    seqs1, seqs_len1 = make_seq_from_tick(infect=params1["A"].numpy(),
        w=[w], n_nodes=len(params1["mu"]),mu=params1["mu"].numpy(), max_time = max_time, num_seq =num_seq)
    seqs2, seqs_len2 = make_seq_from_tick(infect=params2["A"].numpy(),
        w=[w], n_nodes=len(params2["mu"]), mu=params2["mu"].numpy(), max_time = max_time, num_seq =num_seq)
    for n in range(len(seqs2)):
        seqs2[n]['ci'] = seqs2[n]['ci'] + dim
    seqs_train = seqs1[:-100] + seqs2[:-100]
    seqs_len_train = seqs_len1[:-100] + seqs_len2[:-100]
    seqs_test = seqs1[-100:] + seqs2[-100:]
    seqs_len_test = seqs_len1[-100:] + seqs_len2[-100:]
    return {'train':seqs_train, 'test':seqs_test}, [params1, params2], pmat


#define how to get event data
class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, sequences: List[Dict]):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """

        self.time = [seq['ti'].tolist() for seq in sequences]
        self.time_gap = [seq['time_since_last_event'].tolist() for seq in sequences]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [(seq['ci'] + 1).tolist() for seq in sequences]
        self.length = len(sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    return time, time_gap, event_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl




def prepare_dataloader(path):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
            num_types = [data['pmat'].shape[0],data['pmat'].shape[1]]

            return data, num_types

    # print('[Info] Loading train data...')
    data, num_types = load_data(path)
    params1 = data['params1']
    params2 = data['params2']
    pmat = data['pmat']

    return params1,params2,pmat

if __name__=='__main__':#30 45 10
    # data_dict, [params1, params2], pmat=generate_synthetic_tpps(dim= 100,num_seq= 2000,max_time= 10,
    # thres= 0.5, w= 0.1)
    max_time=20
    num_seq=2000
    params1, params2, pmat=prepare_dataloader('./Joint_THP/exp_50_50_2000_45.pkl')
    data_dict, [params1, params2]=generate_synthetic_tpps_from_graph_data(params1, params2, dim= 100,num_seq= num_seq,max_time= max_time,
    thres= 0.5, w= 0.1)
    save_pkl(data_dict, params1, params2, pmat, max_time, num_seq, output_dir='.',)