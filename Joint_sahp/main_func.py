import argparse
import datetime
import glob
import os
import pickle
import numpy as np
import time

import torch
from torch import autograd

from utils.load_synth_data import process_loaded_sequences
from train_functions.train_sahp import make_model, train_eval_sahp

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_LEARN_RATE = 5e-5

parser = argparse.ArgumentParser(description="Train the models.")
parser.add_argument('-e', '--epochs', type=int, default = 1000,
                    help='number of epochs.')
parser.add_argument('-b', '--batch', type=int,
                    dest='batch_size', default=DEFAULT_BATCH_SIZE,
                    help='batch size. (default: {})'.format(DEFAULT_BATCH_SIZE))
parser.add_argument('--lr', default=DEFAULT_LEARN_RATE, type=float,
                    help="set the optimizer learning rate. (default {})".format(DEFAULT_LEARN_RATE))
parser.add_argument('--hidden', type=int,
                    dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                    help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
parser.add_argument('--d-model', type=int, default=DEFAULT_HIDDEN_SIZE)
parser.add_argument('--atten-heads', type=int, default=8)
parser.add_argument('--pe', type=str,default='add',help='concat, add')
parser.add_argument('--nLayers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train-ratio', type=float, default=0.8,
                    help='override the size of the training dataset.')
parser.add_argument('--lambda-l2', type=float, default=3e-4,
                    help='regularization loss.')
parser.add_argument('--dev-ratio', type=float, default=0.1,
                    help='override the size of the dev dataset.')
parser.add_argument('--early-stop-threshold', type=float, default=1e-2,
                    help='early_stop_threshold')
parser.add_argument('--log-dir', type=str,
                    dest='log_dir', default='logs',
                    help="training logs target directory.")
parser.add_argument('--save_model', default=False,
                    help="do not save the models state dict and loss history.")
parser.add_argument('--bias', default=False,
                    help="use bias on the activation (intensity) layer.")
parser.add_argument('--samples', default=10,
                    help="number of samples in the integral.")
parser.add_argument('-m', '--model', default='sahp',
                    type=str, choices=['sahp'],
                    help='choose which models to train.')
parser.add_argument('-t', '--task', type=str, default='retweet',
                    help = 'task type')
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

SYNTH_DATA_FILES = glob.glob("../data/ER_10_90/weighted_directed/exp_10d_*.pkl")
ALIGNED_DATA_FILE = "../data/ER_10_90/pairs_0.2_directed.npz"

# SYNTH_DATA_FILES = glob.glob("../data/hawkes_phone-email/*.pkl")
# ALIGNED_DATA_FILE = "../data/hawkes_phone-email/phone-email_0.1.npz"

TYPE_SIZE_DICT = {'synthetic':(10, 10), 'phone-email':(1000, 1003)}
SYNTHETIC_TASKS = ['synthetic', 'phone-email']

start_time = time.time()

if __name__ == '__main__':
    cuda_num = 'cuda:{}'.format(args.cuda)
    device = torch.device(cuda_num if USE_CUDA else 'cpu')
    print("Training on device {}".format(device))

    process_dim = TYPE_SIZE_DICT[args.task]
    print("Loading {} and {}-dimensional process.".format(process_dim[0], process_dim[1]), end=' \n')

    if args.task in SYNTHETIC_TASKS:
        print("Available files:")
        for i, s in enumerate(SYNTH_DATA_FILES):
            print("{:<8}{:<8}".format(i, s))

        load_aligned_data = np.load(ALIGNED_DATA_FILE)
        train_pairs = torch.LongTensor(load_aligned_data['train_pairs'] / 1.0)
        test_pairs = torch.LongTensor(load_aligned_data['test_pairs'] / 1.0)
        # gnd_pairs = load_aligned_data['gnd']

        tmax_list = []
        train_seq_times_list = []
        train_seq_types_list = []
        train_seq_lengths_list = []
        dev_seq_times_list = []
        dev_seq_types_list = []
        dev_seq_lengths_list = []
        test_seq_times_list = []
        test_seq_types_list = []
        test_seq_lengths_list = []
        max_seq_length_list = []

        for i, synth_data_file in enumerate(SYNTH_DATA_FILES):
            print(synth_data_file)

            with open(synth_data_file, 'rb') as f:
                loaded_hawkes_data = pickle.load(f)
            
            mu = loaded_hawkes_data['mu']
            alpha = loaded_hawkes_data['alpha']
            decay = loaded_hawkes_data['decay']
            tmax = loaded_hawkes_data['tmax']
            print("Simulated Hawkes process parameters:")
            for label, val in [("mu", mu), ("alpha", alpha), ("decay", decay), ("tmax", tmax)]:
                print("{:<20}{}".format(label, val))

            seq_times, seq_types, seq_lengths, _ = process_loaded_sequences(loaded_hawkes_data, process_dim[i])

            seq_times = seq_times.to(device)
            seq_types = seq_types.to(device)
            seq_lengths = seq_lengths.to(device)

            total_sample_size = seq_times.size(0)
            print("Total sample size: {}".format(total_sample_size))

            train_ratio = args.train_ratio
            train_size = int(train_ratio * total_sample_size)
            dev_ratio = args.dev_ratio
            dev_size = int(dev_ratio * total_sample_size)
            print("Train sample size: {:}/{:}".format(train_size, total_sample_size))
            print("Dev sample size: {:}/{:}".format(dev_size, total_sample_size))

            # Define training data
            train_seq_times = seq_times[:train_size]
            train_seq_types = seq_types[:train_size]
            train_seq_lengths = seq_lengths[:train_size]
            print("No. of event tokens in training subset:", train_seq_lengths.sum())

            # Define development data
            dev_seq_times = seq_times[train_size:]#train_size+dev_size
            dev_seq_types = seq_types[train_size:]
            dev_seq_lengths = seq_lengths[train_size:]
            print("No. of event tokens in development subset:", dev_seq_lengths.sum())

            test_seq_times = dev_seq_times
            test_seq_types = dev_seq_types
            test_seq_lengths = dev_seq_lengths
            print("No. of event tokens in test subset:", test_seq_lengths.sum())

            ## sequence length
            train_seq_lengths, reorder_indices_train = train_seq_lengths.sort(descending=True)
            # # Reorder by descending sequence length
            train_seq_times = train_seq_times[reorder_indices_train]
            train_seq_types = train_seq_types[reorder_indices_train]
            #
            dev_seq_lengths, reorder_indices_dev = dev_seq_lengths.sort(descending=True)
            # # Reorder by descending sequence length
            dev_seq_times = dev_seq_times[reorder_indices_dev]
            dev_seq_types = dev_seq_types[reorder_indices_dev]

            test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
            # # Reorder by descending sequence length
            test_seq_times = test_seq_times[reorder_indices_test]
            test_seq_types = test_seq_types[reorder_indices_test]

            max_sequence_length = max(train_seq_lengths[0], dev_seq_lengths[0], test_seq_lengths[0])
            print('max_sequence_length: {}'.format(max_sequence_length))

            tmax_list.append(tmax)
            train_seq_times_list.append(train_seq_times)
            train_seq_types_list.append(train_seq_types)
            train_seq_lengths_list.append(train_seq_lengths)
            dev_seq_times_list.append(dev_seq_times)
            dev_seq_types_list.append(dev_seq_types)
            dev_seq_lengths_list.append(dev_seq_lengths)
            test_seq_times_list.append(test_seq_times)
            test_seq_types_list.append(test_seq_types)
            test_seq_lengths_list.append(test_seq_lengths)
            max_seq_length_list.append(max_sequence_length)
            print('')
            # print("{}: max len {}".format(i, max_sequence_length))

    else:
        exit()
    

    MODEL_TOKEN = args.model
    print("Chose models {}".format(MODEL_TOKEN))
    hidden_size = args.hidden_size
    print("Hidden size: {}".format(hidden_size))
    learning_rate = args.lr
    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    model = None
    if MODEL_TOKEN == 'sahp':
        # with autograd.detect_anomaly():
            params = args, process_dim, device, tmax_list, max_seq_length_list, \
                     train_seq_times_list, train_seq_types_list, train_seq_lengths_list, \
                     dev_seq_times_list, dev_seq_types_list, dev_seq_lengths_list, \
                     test_seq_times_list, test_seq_types_list, test_seq_lengths_list, \
                     BATCH_SIZE, EPOCHS, USE_CUDA, train_pairs, test_pairs
            model = train_eval_sahp(params)

    else:
        exit()


    if args.save_model:
        # Model file dump
        SAVED_MODELS_PATH = os.path.abspath('saved_models')
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        # print("Saved models directory: {}".format(SAVED_MODELS_PATH))

        date_format = "%Y%m%d-%H%M%S"
        now_timestamp = datetime.datetime.now().strftime(date_format)
        extra_tag = "{}".format(args.task)
        filename_base = "{}-{}_hidden{}-{}".format(
            MODEL_TOKEN, extra_tag,
            hidden_size, now_timestamp)
        from utils.save_model import save_model
        save_model(model, SYNTH_DATA_FILES, extra_tag,
                   hidden_size, now_timestamp, MODEL_TOKEN)

    print('Done! time elapsed %.2f sec for %d epoches' % (time.time() - start_time, EPOCHS))

