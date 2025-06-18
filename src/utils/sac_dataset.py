import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import namedtuple, deque
import random

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

Transition = namedtuple('Transition',
                        ('state1', 'state2', 'mode', 'action', 'reward', 'done', 'next_state1', 'next_state2', 'next_mode'))

class SACDataset(Dataset):
    def __init__(self, n_que:int = 1000):
        """Dataset class for SAC.

        Args:
            n_que (int): size of que.
        """
        # Dataset
        self.data_list = deque([], maxlen=n_que)
        
    def shuffle_k_dimension(self, tensor_2d):
        """
        tensor_2d: Tensor of shape [k, feature_dim].
        Randomly shuffle the k dimension.
        """
        k = tensor_2d.shape[0]
        perm = torch.randperm(k)  # k 차원 랜덤 인덱스 생성
        return tensor_2d[perm, :]

    def push(self, state1, state2, mode, action, reward, done, next_state1, next_state2, next_mode):
        # ✅ push할 때 state2, next_state2의 k 차원 셔플 적용
        state2 = self.shuffle_k_dimension(state2)
        next_state2 = self.shuffle_k_dimension(next_state2)

        self.data_list.append(Transition(state1, state2, mode, action, reward, done, next_state1, next_state2, next_mode))

    def sample(self, batch_size):
        samples = random.sample(self.data_list, batch_size)
        batch = Transition(*zip(*samples))

        return torch.cat(batch.state1).detach().to(device=device), \
               pad_sequence(batch.state2, batch_first=True, padding_value=0).transpose(1, 2).to(device=device), \
               torch.cat(batch.mode).detach().to(device=device), \
               torch.cat(batch.action).detach().to(device=device), \
               torch.cat(batch.reward).detach().to(device=device), \
               torch.cat(batch.done).detach().to(device=device), \
               torch.cat(batch.next_state1).detach().to(device=device), \
               pad_sequence(batch.next_state2, batch_first=True, padding_value=0).transpose(1, 2).to(device=device), \
               torch.cat(batch.next_mode).detach().to(device=device)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == '__main__':
    dataset = SACDataset()