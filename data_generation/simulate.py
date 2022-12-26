import pickle
import numpy as np
import torch
from torch.distributions import categorical, exponential, uniform


def save_pkl(process_idx, info, seqs, seqs_len, output_dir='.'):
    data = {}
    mu, alpha, decay, tmax = info
    data['mu'] = list(mu)
    data['alpha'] = np.array(alpha)
    data['decay'] = decay
    data['tmax'] = tmax

    data['timestamps'] = tuple(np.array(seq['ti']) for seq in seqs)
    data['types'] = tuple(np.array(seq['ci']) for seq in seqs)
    data['lengths'] = seqs_len

    with open('{}/exp_{}d_{}.pkl'.format(output_dir, mu.shape[0], process_idx), 'wb') as f:
        pickle.dump(data, f)
    return


def simulate_ogata_thinning(mu, infect, num_seq: int, max_time: float = 5.0, w: float = 1.0):
    
    def _conditional_intensity_vector(seq, t, infect, mu, w):
        lambda_t = torch.clone(mu)

        if seq['ti'] is not None:
            dt = t - seq['ti']
            gt = w * torch.exp(-w * dt)
            gt[dt < 0] = 0
            lambda_t += torch.sum(infect[:, seq['ci']] * gt, dim=1)
 
        return lambda_t

    seqs = []
    seqs_len = []
    for n in range(num_seq):
        sequence = {'ti': None, #event_times
                    'ci': None, #evnet_type
                    'time_since_last_event': None
                    } 

        t = 0.0
        lambda_t = torch.clone(mu)
        lambda_all = torch.sum(lambda_t) 

        exp_dist = exponential.Exponential(rate=lambda_all)
        unif_dist = uniform.Uniform(low=0.0, high=1.0)

        duration = 0.0
        while t < max_time:
            s = exp_dist.sample()
            u = unif_dist.sample()
            lambda_ts = _conditional_intensity_vector(seq=sequence, t=t+s, infect=infect, mu=mu, w=w)
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
                    sequence['time_since_last_event'] = torch.cat([sequence['time_since_last_event'], duration_tensor], dim=0)
                
                duration = torch.zeros(1)
                
            exp_dist = exponential.Exponential(rate=lambda_ts_all)

        if sequence['ci'] is not None:
            seqs_len.append(len(sequence['ci']))
            seqs.append(sequence)
        
    print('Seqs_len: ', seqs_len)
    return seqs, seqs_len
    
    
def make_seq(process_idx, data, num_seq, max_time, w, output_dir='.'):
    data = torch.Tensor(data)
    event_type = data.shape[0]
    mu = torch.ones(event_type, dtype=torch.float32) / event_type
    # deg = torch.sum(data, dim=1)
    # mu = torch.where(deg > 0, deg, torch.tensor(eps, dtype=torch.float32)) / torch.sum(data)
    # mu = torch.zeros(event_type, dtype=torch.float32)
    # mu[nodes] = 1.0
    infect = data / torch.linalg.svdvals(data)[0]
    # infect = data / torch.sum(data, dim=1, keepdim=True)
    # print('SVD before: ', torch.linalg.svdvals(data)[0])
    # print('SVD after: ', torch.linalg.svdvals(infect)[0])
    # infect = data / 5
    sequences, seqs_len = simulate_ogata_thinning(mu, infect, num_seq, max_time, w)
    info = (mu, infect, w, max_time)
    save_pkl(process_idx, info, sequences, seqs_len, output_dir=output_dir)
    return sequences, seqs_len