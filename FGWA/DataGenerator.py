# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti
from typing import List, Tuple
import numpy as np

def data_generator(baseline, decays, adjacency, end_time, n_realizations, seed=1039)-> List:


    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=seed)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

    multi.end_time = [end_time for i in range(n_realizations)]
    multi.simulate()

    return multi.timestamps


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def events_merge(events: List)->List:
    """
     Merges events in a mulitivariate hawkes process into  a single timeseries with different marks
    :param events: Multivariate Hawkes
    :return: Merges Multivariate Hawkes into a single event process with
    Multiple types
    #Todo improvee the typing in the arguments
    """
    events_type = []
    for i, e in enumerate(events):
        events_type = events_type + [[t, i] for t in e]
    events_type = sorted(events_type, key=lambda x: x[0])
    return events_type


def batch_events_merge(N_events: List) -> List:
    """

    Merges batch of events in a mulitivariate hawkes process into  a single timeseries with different marks

    """
    N_events_type = []
    for n, events in enumerate(N_events):
        events_type = events_merge(events)
        events_type = np.array(events_type)
        N_events_type.append(events_type)
    return N_events_type



def Event_batching(N_events_type: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    It recieves merged batch of events output of batch_events_merge
    The outputs the padded events, event_type along with corresponding masks
    :param N_events_type: List
    :return: padded event_times, event_type, masking
    """
    assert  len(N_events_type[0].shape) == 2


    max_length = max( [events.shape[0] for events in N_events_type])
    N = len(N_events_type)
    N_events_pad = np.zeros((N, max_length))
    N_events_type_pad = np.zeros((N, max_length))
    N_events_mask = np.zeros((N, max_length))

    for n in range(N):
        events_type = N_events_type[n]
        events_length = len(events_type)
        N_events_pad[n, :events_length] = events_type[:, 0].reshape(1, -1)
        N_events_type_pad[n, :events_length] = events_type[:, 1].reshape(1, -1)
        N_events_mask[n, :events_length] = np.ones_like(events_type[:, 0])

    return N_events_pad, N_events_type_pad, N_events_mask

def events_per_type(N_events: List, n_types: int) -> List :
    """
    It recievest multivariate hawkes process input return a list with element contain TPP of an event type
    :param N_events:
    :param n_types:
    :return: events per type
    """
    events_type = [[] for i in range(n_types)]
    for n, events in enumerate(N_events):
        for i in range(n_types):
            events_type[i].append(events[i])
    return events_type

def Batching_Events_pertype(event_type: List , n_types: int) :

    """
     It recievest a list with element contain TPP of an event type
     pad each of them to equal length and return it along with its mask
    :param event_type:
    :param n_types:
    :return:
    """
    max_length = 0
    for event_type_i in event_type:
        event_length = max([len(events) for events in event_type_i])
        if max_length < event_length:
            max_length = event_length

    output = []
    for event_type_i in event_type:
        #max_length = max([len(events) for events in event_type_i])
        N  = len(event_type_i)
        N_events_pad = np.zeros((N, max_length))
        N_events_mask = np.zeros((N, max_length))

        for n in range(N):
            events = event_type_i[n]
            events_length = len(events)
            N_events_pad[n, :events_length] = events
            N_events_mask[n, :events_length] = 1

        output.append((N_events_pad, N_events_mask))

    return output



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    N_events = data_generator()
    events_process = batch_events_merge(N_events)
    print(len(events_process))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
