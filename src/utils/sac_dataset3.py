import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple, deque
import random

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

Transition = namedtuple('Transition',
                        ('state1', 'state2', 'action', 'reward', 'next_state1', 'next_state2'))

class SACDataset(Dataset):
    def __init__(self, n_que:int = 1000):
        """Dataset class for SAC.

        Args:
            n_que (int): size of que.
        """
        # Dataset
        self.data_list = deque([], maxlen=n_que)
        self.pre_data = deque([], maxlen=n_que)

    def add_dataset(self, *args):
        self.pre_data.append(Transition(*args))

    def push(self, *args):
        self.data_list.append(Transition(*args))

    def sample(self, batch_size):

        len_data_list = len(self.data_list)
        len_pre_data = len(self.pre_data)
        total_length = len_data_list + len_pre_data
        num_samples_data_list = int(batch_size * len_data_list / total_length)
        num_samples_pre_data = batch_size - num_samples_data_list

        samples_data_list = random.sample(self.data_list, num_samples_data_list)
        samples_pre_data = random.sample(self.pre_data, num_samples_pre_data)

        samples = samples_data_list + samples_pre_data
        batch = Transition(*zip(*samples))

        return torch.cat(batch.state1).detach().to(device=device), \
               pad_sequence(batch.state2, batch_first=True, padding_value=0).transpose(1, 2), \
               torch.cat(batch.action).detach(), \
               torch.cat(batch.reward).detach(), \
               torch.cat(batch.next_state1).detach().to(device=device), \
               pad_sequence(batch.next_state2, batch_first=True, padding_value=0).transpose(1, 2), \

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __add__(self, other):
        
        return super().__add__(other)

if __name__ == '__main__':
    dataset = SACDataset()