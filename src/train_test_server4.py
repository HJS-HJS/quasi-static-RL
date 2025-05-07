'''
SAC (Soft Actor Critic)
- continous action
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time

so_file_path = os.path.abspath("../cpp/15")
sys.path.append(so_file_path)

from utils.simulation_server2 import DishSimulation

from utils.sac_dataset_cpp_linear import SACDataset
from utils.utils           import *


## Parameters
# TRAIN           = False
TRAIN           = True
curriculum = 0

FILE_NAME = None

loss = 0.

# Learning frame
FRAME = 8

# Learning Parameters
LEARNING_RATE   = 0.0006 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
TARGET_UPDATE_TAU= 0.01
EPISODES        = 15000   # total episode
ALPHA           = 0.5
LEARNING_RATE_ALPHA= 0.0001
# Memory
MEMORY_CAPACITY = 150000
BATCH_SIZE = 1024
EPOCH_SIZE = 2
# Other
visulaize_step = 50
MAX_STEP = 200         # maximun available step per episode
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/../model/test_server4"
episode_start = 1

# curriculum
curriculum_dictionary = np.array([
    # obs, action_step, target_entropy
    [2, 2, -2],
    [4, 3, -2],
    [5, 4, -2],
    [6, 4, -2],
    [8, 4, -4],
    [9, 4, -4],
])  

sim = DishSimulation(
    visualize=None,
    state="linear",
    action_skip=FRAME,
    )
sim.env.spawn_bias = 0.95

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS1   = 21 #9
N_INPUTS2   = 26 #9
N_OUTPUT    = sim.env.action_space.shape[0] - 1   # 5

total_steps = []
success_rates = []

# Memory
memory = SACDataset(MEMORY_CAPACITY)

def mask_attention_output(attn_output, mask):
    """
    attn_output: [batch, k, hidden_dim]
    mask: [batch, k] (True: 패딩된 부분, False: 실제 장애물)
    
    패딩된 부분을 강제로 0으로 변환
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)  # [batch, k] → [batch, k, 1]
        attn_output = attn_output.masked_fill(mask_expanded, 0.0)
    return attn_output

import torch.nn.functional as F

import torch
import torch.nn as nn

class SelfAttentionObstacle(nn.Module):
    def __init__(self, obs_dim=10, hidden_dim=1024, num_heads=8):
        super(SelfAttentionObstacle, self).__init__()
        self.linear_in = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs, mask=None):
        """
        obs: [batch, k, obs_dim]
        mask: [batch, k] (True for padding)
        pooled: bool, whether to return pooled output or full sequence
        """
        obs = obs.permute(0, 2, 1)  # [batch, obs_dim, k] → [batch, k, obs_dim]
        obs_embed = self.linear_in(obs)

        attn_output, _ = self.attention(obs_embed, obs_embed, obs_embed, key_padding_mask=mask)

        if mask is not None:
            attn_output = mask_attention_output(attn_output, mask)

        out = self.linear_out(attn_output)
        if mask is not None:
            # mask = [batch, k] → [batch, k, 1] for broadcasting
            mask_inv = (~mask).unsqueeze(-1).float()  # 1 for valid, 0 for masked
            sum_feat = (out * mask_inv).sum(dim=1)  # sum only valid features
            count = mask_inv.sum(dim=1).clamp(min=1.0)  # avoid division by 0
            return sum_feat / count  # [batch, hidden_dim]
        else:
            return out.mean(dim=1)  # [batch, hidden_dim]
        
class SelfAttentionObstacle(nn.Module):
    def __init__(self, obs_dim=10, hidden_dim=1024):
        super(SelfAttentionObstacle, self).__init__()
        hidden_dim = int(hidden_dim / 2)
        hidden_dim_half = int(hidden_dim / 2)
        self.mean_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_half),
            nn.ReLU(),
            nn.Linear(hidden_dim_half, hidden_dim_half),
            nn.ReLU(),
            nn.Linear(hidden_dim_half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.max_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim_half),
            nn.ReLU(),
            nn.Linear(hidden_dim_half, hidden_dim_half),
            nn.ReLU(),
            nn.Linear(hidden_dim_half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs):
        obs = obs.permute(0, 2, 1)  # [batch, k, 10]

        valid_mask = (obs.abs().sum(dim=2) != 0)  # 실제 장애물 여부
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [batch, 1]

        mean_obs = self.mean_layer(obs)
        max_obs = self.max_layer(obs)

        # 패딩은 무시하고 평균 계산
        mean_obs_masked = mean_obs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)  # 패딩된 부분을 0으로 만듦
        mean_obs = mean_obs_masked.sum(dim=1) / valid_counts  # [batch, dim]

        max_obs_masked = max_obs.masked_fill(~valid_mask.unsqueeze(-1), -1e9)  # 패딩된 부분을 -1e9으로 만듦
        max_obs = max_obs_masked.max(dim=1)[0] / valid_counts  # [batch, dim]

        return torch.cat([mean_obs, max_obs], dim=1)

class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.self_attention = SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=512)

        self.mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
            ) for _ in range(2)
        ])

        self.std = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
                nn.Softplus(),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, mode):
        mask = (obs.abs().sum(dim=1) == 0)

        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)
        # State branch
        _state = self.layer(state)  # [batch, 1024]

        # Obs self attention
        _obs = self.self_attention(obs, mask)  # [batch, 1024]
        _state = torch.cat([_state, _obs], dim=1)

        if mode.numel() == 1:
            mode_idx = mode.item()
            mu = self.mu[mode_idx](_state)
            std = self.std[mode_idx](_state)
            # _mu = torch.cat([relative_pose, _mu], dim=1)
            # mu = self.mu2[mode_idx](_mu)

        else:
            mu = torch.where(mode.bool(), self.mu[1](_state), self.mu[0](_state))
            std = torch.where(mode.bool(), self.std[1](_state), self.std[0](_state))

            # _mu = torch.cat([relative_pose, _mu], dim=1)
            # mu = torch.where(mode.bool(), self.mu2[1](_mu), self.mu2[0](_mu))
                
        return mu, torch.clamp(std, min=0.05, max=2.0)
    

    def sample_action(self, state, obs, mode):
        mu, std = self.forward(state, obs, mode)

        _cut = curriculum_dictionary[curriculum][1]
        use_idx = {
            2: [0, 1],             # x, y
            3: [0, 1, 3],          # x, y, width (theta는 0으로)
            4: [0, 1, 2, 3],       # x, y, theta, width
        }[_cut]
        
        # Sample continous action
        if mu.dim() == 2:
            distribution = torch.distributions.Normal(mu[:, use_idx], std[:, use_idx])
        else:
            distribution = torch.distributions.Normal(mu[:, :, use_idx], std[:, :, use_idx])

        u = distribution.rsample()
        log_prob_continue = distribution.log_prob(u)

        # Enforce action bounds [-1., 1.]
        action_countinue = torch.tanh(u)
        log_prob_continue -= torch.log(1 - torch.tanh(u).pow(2) + 1e-6)

        # Curriculum
        if mu.dim() == 2:
            if _cut == 2:
                theta = torch.zeros_like(action_countinue[:, :1])  # fixed theta
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 1.0).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, theta, width], dim=1)
            
            elif _cut == 3:
                xy = action_countinue[:, :2]
                width = action_countinue[:, 2:3]
                theta = torch.zeros_like(width)
                action_countinue = torch.cat([xy, theta, width], dim=1)

            else:
                pass
        else:
            if _cut == 2:
                theta = torch.zeros_like(action_countinue[:, :, :1])  # fixed theta
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 1.0).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, theta, width], dim=1)
            
            elif _cut == 3:
                xy = action_countinue[:, :2]
                width = action_countinue[:, 2:3]
                theta = torch.zeros_like(width)
                action_countinue = torch.cat([xy, theta, width], dim=1)

            else:
                pass

        return action_countinue, log_prob_continue.sum(dim = 1)


class QNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(QNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.action_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_action, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            ) for _ in range(2)
        ])

        self.self_attention = SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=512)

        self.layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, action, mode):
        mask = (obs.abs().sum(dim=1) == 0)

        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)

        # Run
        _state = self.state_layer(state)  # [batch, 1024]
        _action = torch.where(mode.bool(), self.action_layer[0](action), self.action_layer[1](action))  # [batch, 1024]

        # Obs self attention
        _obs = self.self_attention(obs, mask)  # [batch, 1024]
        fusion_input = torch.cat([_state, _obs, _action], dim=1)  # [batch, 768]

        _q = torch.where(mode.bool(), self.layer[1](fusion_input), self.layer[0](fusion_input))
        # _q = torch.cat([relative_pose, _q], dim=1)

        # return torch.where(mode.bool(), self.layer2[1](_q), self.layer2[0](_q))
        return _q
            
    def train(self, target, state, obs, action, mode, optimizer):
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(self.forward(state, obs, action, mode) , target)
        optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        optimizer.step()

    def update(self, target_net:nn.Module):
        for target_param, param in zip(target_net.parameters(), self.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TARGET_UPDATE_TAU) + param.data * TARGET_UPDATE_TAU)


# Initialize network
actor_net = ActorNetwork(N_INPUTS1, N_INPUTS2, N_OUTPUT).to(device)
q1_net = QNetwork(N_INPUTS1, N_INPUTS2, N_OUTPUT).to(device)
q2_net = QNetwork(N_INPUTS1, N_INPUTS2, N_OUTPUT).to(device)
target_q1_net = QNetwork(N_INPUTS1, N_INPUTS2, N_OUTPUT).to(device)
target_q2_net = QNetwork(N_INPUTS1, N_INPUTS2, N_OUTPUT).to(device)
alpha = torch.tensor(np.log(ALPHA))
alpha.requires_grad = True

target_q1_net.load_state_dict(q1_net.state_dict())
target_q2_net.load_state_dict(q2_net.state_dict())

try:
    actor_net, q1_net, q2_net, target_q1_net, target_q2_net = load_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                    , SAVE_DIR, 
                    ["actor", "q1", "q2", "target_q1", "target_q2"]
                    , FILE_NAME
                    )
    alpha = load_tensor(alpha, SAVE_DIR, "alpha", FILE_NAME)
    alpha.requires_grad = True
    total_steps = load_numpy(total_steps, SAVE_DIR, "total_steps", FILE_NAME)
    success_rates  = load_numpy(success_rates, SAVE_DIR, "success_rates", FILE_NAME)
    episode_start = load_episode(SAVE_DIR, FILE_NAME) + 1
except:
    episode_start = 1

# Optimizer
actor_optimizer   = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE)
q1_optimizer = torch.optim.AdamW(q1_net.parameters(), lr=LEARNING_RATE)
q2_optimizer = torch.optim.AdamW(q2_net.parameters(), lr=LEARNING_RATE)
alpha_optimizer   = torch.optim.AdamW([alpha], lr=LEARNING_RATE_ALPHA)

def optimize_model(batch):

    s1, s2, m, a, r, d, next_s1, next_s2, next_m = batch

    # Calculate 
    with torch.no_grad():
        next_a, next_logprob = actor_net.sample_action(next_s1, next_s2, next_m)
        next_entropy = -alpha.exp() * next_logprob

        next_q1 = target_q1_net(next_s1, next_s2, next_a, next_m)
        next_q2 = target_q2_net(next_s1, next_s2, next_a, next_m)

        next_min_q = torch.min(torch.cat([next_q1, next_q2], dim=1), 1)[0]
        target = r + (1 - d) * DISCOUNT_FACTOR * (next_min_q + next_entropy).unsqueeze(1)

        mode = next_m.long().view(-1, 1)
        target += torch.where(mode.bool(), 0.0, DISCOUNT_FACTOR * (next_min_q + next_entropy).unsqueeze(1))  # [batch, 1024]

    q1_net.train(target, s1, s2, a, m, q1_optimizer)
    q2_net.train(target, s1, s2, a, m, q2_optimizer)

    action, logprob_batch = actor_net.sample_action(s1, s2, m)
    q1 = q1_net(s1, s2, action, m)
    q2 = q2_net(s1, s2, action, m)
    
    min_q = torch.min(torch.cat([q1, q2],dim=1), 1)[0]
    actor_loss = (alpha.exp() * logprob_batch - min_q) # GPT
    actor_optimizer.zero_grad()
    actor_loss.mean().backward()
    global loss
    loss = actor_loss.mean().item()
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=0.5)

    actor_optimizer.step()

    alpha_loss = (alpha.exp() * (-logprob_batch.detach() - curriculum_dictionary[curriculum][2].astype(float))).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    q1_net.update(target_q1_net)
    q2_net.update(target_q2_net)

step_done_set = []
prev_curriculum_episode = episode_start
if TRAIN:
    for episode in range(episode_start, EPISODES + 1):
        limit = 0
        if (episode // 4) % 4 == 2:   _num = curriculum_dictionary[curriculum][0]
        elif (episode // 4) % 4 == 3: _num = curriculum_dictionary[curriculum][0]
        elif (episode // 4) % 4 == 0: _num = curriculum_dictionary[curriculum][0] 
        elif (episode // 4) % 4 == 1:
            if episode % 2 == 0: _num = max(2, curriculum_dictionary[curriculum][0] // 2)
            else:                _num = max(2, curriculum_dictionary[curriculum][0] // 4)
        limit = _num - 1
        
        obs_num = 0

        while obs_num == 0:
            state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num)
            state_curr1, state_curr2 = state_curr
            valid_counts = np.sum(np.abs(state_curr2), axis = 1)  # [batch]
            obs_num = len(valid_counts) - len(np.where(valid_counts == 0)[0])
            if obs_num < limit:
                obs_num = 0

        state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)

        # Running one episode
        total_reward = 0.0
        start_time = time.time()
        mode_change_step = 0

        for step in range(1, MAX_STEP + 1):
            # 1. Get action from policy network
            with torch.no_grad():
                action, logprob = actor_net.sample_action(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                action = action.squeeze().cpu().numpy()

            if step > 4 and mode == 0:
                rand = (2 * np.random.random(action.size) - 1) * (step / MAX_STEP)
                rand[:2] /= 2
                action = np.clip(action + rand, -0.95, 0.95).astype(np.float32)

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done, mode_next = sim.env.step(action, mode)

            state_next1, state_next2 = state_next
            total_reward += reward

            # Check simulation break
            if reward < -500:
                done = True
                break

            if done != 0: _done = 1
            else: _done = 0
            if mode_next == 0:
                mode_change_step = step
                _done = 1
                
            # 3. Update state
            state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
            state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)

            # 4. Save data
            memory.push(
                state_curr1.to(device=torch.device('cpu')),
                state_curr2.T,
                torch.tensor([mode], device=device).unsqueeze(0),
                torch.tensor(action, device=device).unsqueeze(0),
                torch.tensor([reward], device=device).unsqueeze(0),
                torch.tensor([_done], device=device).unsqueeze(0),
                state_next1.to(device=torch.device('cpu')),
                state_next2.T,
                torch.tensor([mode_next], device=device).unsqueeze(0),
            )

            # 5. Update state                
            if mode_next != 0:
                state_curr1 = state_next1
                state_curr2 = state_next2
                mode = mode_next

            # 6. Learning
            if (len(memory) > BATCH_SIZE):
                for _ in range(EPOCH_SIZE):
                    optimize_model(memory.sample(BATCH_SIZE))
            if done: 
                break
        
        ## Episode is finished
        print("\t#", episode, "\t", step, "\t{:.2f}\t{:.2f}s\te {:.4f}".format(total_reward, time.time() - start_time, alpha.exp().tolist()), "\tmode changed: ", mode_change_step, "\tloss: ", loss, "\tobs: ", obs_num)
        if (reward < 1): step = MAX_STEP
        
        # Save episode reward
        step_done_set.append(step)
        if episode % visulaize_step == 0:
            save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                        , SAVE_DIR, 
                        ["actor", "q1", "q2", "target_q1", "target_q2"]
                        , episode
                        )
            success_rate = 100 * len(np.where(np.array(step_done_set) != MAX_STEP)[0]) / visulaize_step
            print("#{}: {}\t{:.1f}%\t{}".format(episode, np.mean(step_done_set).astype(int), success_rate, curriculum))

            save_tensor(alpha, SAVE_DIR, "alpha", episode)
            total_steps = np.hstack((total_steps, np.mean(step_done_set)))
            success_rates = np.hstack((success_rates, success_rate * 2))
            save_numpy(total_steps, SAVE_DIR, "total_steps", episode)
            save_numpy(success_rates, SAVE_DIR, "success_rates", episode)
            live_plots([total_steps, success_rates], visulaize_step)
            step_done_set = []
        
            if (success_rate > 50 and episode >= prev_curriculum_episode + 1500) or (success_rate > 80 and episode >= prev_curriculum_episode + 500) or (success_rate > 89.99 and episode >= prev_curriculum_episode + 200):
                if curriculum != len(curriculum_dictionary) - 1:
                    curriculum = min(curriculum + 1, len(curriculum_dictionary) - 1)
                    prev_curriculum_episode = episode
                    alpha = torch.tensor(np.log(ALPHA))
                    alpha.requires_grad = True
                    alpha_optimizer   = torch.optim.AdamW([alpha], lr=LEARNING_RATE_ALPHA)
                
    # Turn the sim off
    save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                , SAVE_DIR, 
                ["actor", "q1", "q2", "target_q1", "target_q2"]
                , "save"
                )
    # Show the results
    show_result([total_steps, success_rates], visulaize_step, SAVE_DIR)

else:
    del sim
    sim = DishSimulation(
                        visualize=None,
                        state="linear",
                        # record=True,
                        action_skip=FRAME
                        )
    actor_net = load_model(actor_net, SAVE_DIR, "actor", FILE_NAME)
    q1_net = load_model(q1_net, SAVE_DIR, "q1", FILE_NAME)
    q2_net = load_model(q2_net, SAVE_DIR, "q2", FILE_NAME)
    target_q1_net = load_model(target_q1_net, SAVE_DIR, "target_q1", FILE_NAME)
    target_q2_net = load_model(target_q2_net, SAVE_DIR, "target_q2", FILE_NAME)
    alpha = load_tensor(alpha, SAVE_DIR, "alpha", FILE_NAME)

    dish = np.zeros((15, 3), dtype=int)
    failed_case = np.zeros(3, dtype=int)
    index = np.array(["dish", "try", "success", "failed"])
    idx = 0
    while True: 
        mode_change_step = 0
        while True:
            if idx % 3 == 0:
                _num = curriculum_dictionary[curriculum][0]
            elif idx % 3 == 1:
                _num = curriculum_dictionary[curriculum][0]
            else:
                _num = curriculum_dictionary[curriculum][0] // 2
            state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num)
            limit = _num - 2
            state_curr1, state_curr2 = state_curr
            obs_num = np.sum(np.any(state_curr2, axis=1))

            if obs_num < limit: continue
            idx += 1
            obs_num -= 1
            if np.sum(dish[obs_num,0]) < 200:
                break


        state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)

        # Running one episode
        for step in range(1, MAX_STEP + 1):
            # 1. Get action from policy network
            with torch.no_grad():
                action, logprob = actor_net.sample_action(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                _action = action.squeeze().cpu().numpy()

            if step > 10:
                rand = (2 * np.random.random(_action.size) - 1) * (step / MAX_STEP)
                rand[2:] *= 2
                _action = np.clip(_action + rand, -0.9999, 0.9999)

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done, mode = sim.env.step(_action, mode)

            # print("q1", q1_net(state_curr1, state_curr2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))
            # print("q2", q2_net(state_curr1, state_curr2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))

            if mode == 0:
                mode_change_step = step
            else :
                state_next1, state_next2 = state_next
                state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                state_curr1 = state_next1
                state_curr2 = state_next2
                break
            if done: break

        # Running one episode
        if not done: 
            for step in range(1, MAX_STEP + 1):
                # 1. Get action from policy network
                with torch.no_grad():
                    action, logprob = actor_net.sample_action(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))

                # 2. Run simulation 1 step (Execute action and observe reward)
                state_next, reward, done, mode = sim.env.step(action.squeeze().cpu().numpy(), mode)

                if mode == 0:
                    mode_change_step = step
                else :
                    state_next1, state_next2 = state_next
                    state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                    state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                    state_curr1 = state_next1
                    state_curr2 = state_next2
                # print("reward", reward)
                # print("action", action.squeeze().cpu().numpy())
                # print("q1", q1_net(state_next1, state_next2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))
                # print("q2", q2_net(state_next1, state_next2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))
                if done: break

        dish[obs_num, 0] += 1
        if reward > 0.5: dish[obs_num, 1] += 1
        else:            dish[obs_num, 2] += 1

        if reward == -2.5: failed_case[1] += 1
        elif reward == -2.25: failed_case[2] += 1
        elif reward < 0.5: failed_case[0] += 1

        print("\tEpisode finished")
        print("\t\tStep #{}\tMode changed #{}".format(step, mode_change_step))
        print("\t\tSuccess Rate {:.2f}%\tTry {}\tSuccess {}\tFail {}".format(np.sum(dish[:,1]) / np.sum(dish[:,0]) * 100, np.sum(dish[:,0]), np.sum(dish[:,1]), np.sum(dish[:,2])))
        print("\t\t{}".format(index[0]), end="\t")
        for i in range(15):
            print("#{}".format(i), end="\t")
        print("")
        for j in range(3):
            print("\t\t{}".format(index[j + 1]), end="\t")
            for i in range(15):
                print("{}".format(dish[i, j]), end="\t")
            print("")
        print("\tFailed Case Out: {}\tFailed: {}\tOut: {}".format(failed_case[0], failed_case[1], failed_case[2]))
        time.sleep(1)

# Turn the sim off
sim.env.close()
