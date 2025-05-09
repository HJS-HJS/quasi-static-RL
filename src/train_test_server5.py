'''
SAC (Soft Actor Critic)
- continous action
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import copy
import time

so_file_path = os.path.abspath("../cpp")
sys.path.append(so_file_path)

from utils.simulation_server3 import DishSimulation

from utils.sac_dataset_cpp_linear import SACDataset
from utils.utils           import *

## Parameters
# TRAIN           = False
TRAIN           = True
use_data        = False

# show_mu = not TRAIN
# show_mu = True
show_mu = False

# HER = True
HER = False

# FILE_NAME = "6200"
FILE_NAME = None

loss = 0.

# Learning frame
FRAME = 16
# Learning Parameters
LEARNING_RATE   = 0.0004 # optimizer
DISCOUNT_FACTOR = 0.95   # gamma
TARGET_UPDATE_TAU= 0.01
EPISODES        = 15000   # total episode
TARGET_ENTROPY  = -0.0
ALPHA           = 0.5
LEARNING_RATE_ALPHA= 0.005
# Memory
MEMORY_CAPACITY = 60000
BATCH_SIZE = 256
EPOCH_SIZE = 1
# Other
visulaize_step = 50
MAX_STEP = 150         # maximun available step per episode
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/../model/test_server5"
episode_start = 1

sim = DishSimulation(
    visualize=None,
    state="linear",
    action_skip=FRAME,
    )
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS1   = 21 #9
N_INPUTS2   = 19 #9
# N_INPUTS1   = 17 #10
# N_INPUTS2   = 11 #10
# N_INPUTS1   = 22
# N_INPUTS2   = 19
# N_INPUTS1   = 21
# N_INPUTS2   = 17
N_OUTPUT    = sim.env.action_space.shape[0] - 1   # 5

total_steps = []

# Memory
memory = SACDataset(MEMORY_CAPACITY)

if use_data:
    for action, state, mode, (state_next, reward, done, mode_next) in sim.replay_video():\

        state_curr1, state_curr2 = state
        state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)
        state_next1, state_next2 = state_next
        state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
        state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)

        if mode < 0:      mode = 0
        else:             mode = 1

        memory.push(
            state_curr1.to(device=torch.device('cpu')),
            state_next2.T,
            torch.tensor([mode], device=device).unsqueeze(0),
            torch.tensor(action, device=device).unsqueeze(0),
            torch.tensor([reward], device=device).unsqueeze(0),
            torch.tensor([done], device=device).unsqueeze(0),
            state_next1.to(device=torch.device('cpu')),
            state_next2.T,
            torch.tensor([mode_next], device=device).unsqueeze(0),
        )
    print("Data Loaded #", len(memory))

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

class SelfAttentionObstacle(nn.Module):
    def __init__(self, obs_dim=10, hidden_dim=1024, num_heads=8):
        super(SelfAttentionObstacle, self).__init__()
        self.linear_in = nn.Linear(obs_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs, mask=None):
        obs = obs.permute(0, 2, 1)  # [batch, k, 10]
        obs_embed = self.linear_in(obs)  # [batch, k, hidden_dim]

        # Self-Attention 적용
        attn_output, attn_weights = self.attention(obs_embed, obs_embed, obs_embed, key_padding_mask=mask)
        if show_mu:
            print("Attention Weights Mean:\t", torch.mean(attn_weights, dim=1).squeeze().cpu().numpy())

        attn_output = self.norm1(attn_output + obs_embed)
        attn_output = self.ffn(attn_output)
        attn_output = self.norm2(attn_output + obs_embed)

        # 패딩된 부분을 0으로 설정
        attn_output = mask_attention_output(attn_output, mask)

        # **Masked Weighted Sum**
        valid_mask = ~mask  # [batch, k] (True: 실제 장애물, False: 패딩)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # 실제 장애물 개수
        attn_weights_masked = attn_weights * valid_mask.unsqueeze(1)  # [batch, 1, k]
        attn_weights_mean = attn_weights_masked.sum(dim=2, keepdim=True) / valid_counts.unsqueeze(1)  # [batch, 1, k]

        # 🚀 **Masked Weighted Sum 적용 (k 축 제거)**
        obs_weighted_sum = torch.sum(attn_weights_mean * attn_output, dim=1)  # [batch, hidden_dim]

        return obs_weighted_sum



class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.self_attention = SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=1024, num_heads=8)

        self.mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, n_action),
            ),
            nn.Sequential(
                nn.Linear(1024 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, n_action),
            ),
        ])

        self.std = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_action),
                nn.Softplus(),
            ),
            nn.Sequential(
                nn.Linear(1024 * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_action),
                nn.Softplus(),
            ),
        ])

    def forward(self, state, obs, mode):
        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)

        # State branch
        # _state = torch.where(mode.bool(), self.layer[0](state), self.layer[0](state))  # [batch, 1024]
        _state = self.layer(state)  # [batch, 1024]

        if show_mu: 
            print("mu:\t", self.mu[0][0].weight.abs().mean().item())
            print("mu:\t", self.mu[1][0].weight.abs().mean().item())

        # Mask
        # Mask if obstacle not exist in each k
        valid_mask = (obs.abs().sum(dim=1) != 0)  # [batch, k]
        mask = (obs.abs().sum(dim=1) == 0)
        valid_mask_expanded = valid_mask.unsqueeze(1)  # [batch, 1, k]
        # Mask if obstacle not existd in every k
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        no_valid = valid_counts.eq(0)  # [batch, 1]

        # Obs self attention
        _obs = self.self_attention(obs, mask)  # [batch, 1024]
        # obs_feat_masked = torch.where(valid_mask_expanded, _obs, torch.full_like(_obs, -1e9))

        # # # max
        # obs_max = torch.max(obs_feat_masked, 2)[0]
        # # # mean
        # valid_counts = valid_counts.expand(-1, obs_feat_masked.size(1))  # 차원 맞춤
        # obs_mean = (obs_feat_masked * valid_mask_expanded).sum(dim=2) / valid_counts  # [batch, 1024]

        # obs_output_max  = torch.where(no_valid, torch.zeros_like(obs_max), obs_max)
        # obs_output_mean= torch.where(no_valid, torch.zeros_like(obs_mean), obs_mean)
    
        # obs_output_max, obs_output_mean = obs_output_max + obs_output_mean * 0.1, obs_output_mean + obs_output_max * 0.1

        # _state = torch.cat([_state, obs_output_max * 2.0, obs_output_mean * 2.0], dim=1)
        _state = torch.cat([_state, _obs], dim=1)

        if mode.numel() == 1:
            mode_idx = mode.item()
            mu = self.mu[mode_idx](_state)
            std = self.std[mode_idx](_state)
        else:
            mu = torch.where(mode.bool(), self.mu[1](_state), self.mu[0](_state))
            std = torch.where(mode.bool(), self.std[1](_state), self.std[0](_state))
                
        return mu, torch.clamp(std, min=0.01, max=2.0)
    

    def sample_action(self, state, obs, mode):
        mu, std = self.forward(state, obs, mode)

        # Sample continous action
        distribution = torch.distributions.Normal(mu, std)
        u = distribution.rsample()
        log_prob_continue = distribution.log_prob(u)

        # Enforce action bounds [-1., 1.]
        action_countinue = torch.tanh(u)
        log_prob_continue -= torch.log(1 - torch.tanh(u).pow(2) + 1e-6)

        if len(mu) == 1 and show_mu:
            print(f"Mu0: {mu[0].tolist()}, Std: {std[0].tolist()}")
            print(f"Alpha: {alpha.exp().item()}")

        return action_countinue, log_prob_continue.mean(dim = 1)

class QNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(QNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(n_state, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.action_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_action, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(n_action, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
            ),
        ])

        self.self_attention = SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=1024, num_heads=8)

        self.layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 * 3, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            ),
            nn.Sequential(
                nn.Linear(1024 * 3, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            ),
        ])

    def forward(self, state, obs, action, mode):
        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)

        # Run
        _state = self.state_layer(state)  # [batch, 1024]
        # _state = torch.where(mode.bool(), self.state_layer[0](state), self.state_layer[0](state))  # [batch, 1024]
        _action = torch.where(mode.bool(), self.action_layer[0](action), self.action_layer[1](action))  # [batch, 1024]

        # Mask
        # Mask if obstacle not exist in each k
        valid_mask = (obs.abs().sum(dim=1) != 0)  # [batch, k]
        mask = (obs.abs().sum(dim=1) == 0)
        valid_mask_expanded = valid_mask.unsqueeze(1)  # [batch, 1, k]
        # Mask if obstacle not existd in every k
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
        no_valid = valid_counts.eq(0)  # [batch, 1]

        # Obs self attention
        _obs = self.self_attention(obs, mask)  # [batch, 1024]
        # obs_feat_masked = torch.where(valid_mask_expanded, _obs, torch.full_like(_obs, -1e9))

        # # max
        # obs_max = torch.max(obs_feat_masked, 2)[0]
        # # mean
        # valid_counts = valid_counts.expand(-1, obs_feat_masked.size(1))  # 차원 맞춤
        # obs_mean = (obs_feat_masked * valid_mask_expanded).sum(dim=2) / valid_counts  # [batch, 1024]

        # obs_output_max  = torch.where(no_valid, torch.zeros_like(obs_max), obs_max)
        # obs_output_mean= torch.where(no_valid, torch.zeros_like(obs_mean), obs_mean)

        # obs_output_max, obs_output_mean = obs_output_max + obs_output_mean * 0.1, obs_output_mean + obs_output_max * 0.1

        # fusion_input = torch.cat([_state, obs_output_max * 2, obs_output_mean * 2, _action], dim=1)  # [batch, 768]
        fusion_input = torch.cat([_state, _obs, _action], dim=1)  # [batch, 768]

        return torch.where(mode.bool(), self.layer[1](fusion_input), self.layer[0](fusion_input))
            
    def train(self, target, state, obs, action, mode, optimizer):
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(self.forward(state, obs, action, mode) , target)
        optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        # loss.sum().backward()
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

    q1_net.train(target, s1, s2, a, m, q1_optimizer)
    q2_net.train(target, s1, s2, a, m, q2_optimizer)

    action, logprob_batch = actor_net.sample_action(s1, s2, m)
    q1 = q1_net(s1, s2, action, m)
    q2 = q2_net(s1, s2, action, m)
    
    min_q = torch.min(torch.cat([q1, q2],dim=1), 1)[0]
    actor_loss = (alpha.exp() * logprob_batch - min_q) # GPT
    actor_optimizer.zero_grad()
    actor_loss.mean().backward()
    # actor_loss.sum().backward()
    # if True:
        # print(f"Actor Loss: {actor_loss.sum().item()}, Alpha: {alpha.exp().item()}")
    global loss
    loss = actor_loss.mean().item()
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=0.5)

    actor_optimizer.step()

    alpha_loss = (alpha.exp() * (-logprob_batch.detach() - TARGET_ENTROPY)).mean()
    # alpha_loss = (alpha.exp() * (-logprob_batch.detach() - TARGET_ENTROPY)).sum()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    q1_net.update(target_q1_net)
    q2_net.update(target_q2_net)

step_done_set = []
if TRAIN:
    for episode in range(episode_start, EPISODES + 1):
        limit = 0
        if (episode // 4) % 4 == 2:
            _num = 15
            limit = 6
        elif (episode // 4) % 4 == 3:
            _num = 15
            limit = 8
        elif (episode // 4) % 4 == 0:
            if episode % 2 == 0:
                _num = 2
                limit = 0
            else:
                _num = 5
                limit = 4
        elif (episode // 4) % 4 == 1:
            if episode % 2 == 0:
                _num = 6
                limit = 5
            else:
                _num = 15
                limit = 7
        
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
                
            if mode == 0 and np.random.random(1) > 0.95:
                action = (np.random.choice([-1.0, 0.0, 1.0], size=4) * np.random.uniform(0.8, 1.0, size=4)).astype(np.float32)

            if step > 4 and mode == 0:
                rand = (2 * np.random.random(action.size) - 1) * (step / MAX_STEP)
                rand[2:] *= 2
                action = np.clip(action + rand, -0.9999, 0.9999).astype(np.float32)

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done, mode_next = sim.env.step(action, mode)

            # Check simulation break
            if reward < -500:
                done = True
                break

            if done != 0: _done = 1
            else: _done = 0
            state_next1, state_next2 = state_next
            if mode_next == 0:
                mode_change_step = step
                _done = 1
            total_reward += reward

            # 3. Update state
            state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
            state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)

            # 4. Save data
            memory.push(
                state_curr1.to(device=device),
                state_curr2.T,
                torch.tensor([mode], device=device).unsqueeze(0),
                torch.tensor(action, device=device).unsqueeze(0),
                torch.tensor([reward], device=device).unsqueeze(0),
                torch.tensor([_done], device=device).unsqueeze(0),
                state_next1.to(device=device),
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
        if (reward < 3): step = MAX_STEP
        
        # Save episode reward
        step_done_set.append(step)
        # Visualize
        # if (len(total_steps) != 0) and (step < min(total_steps)):
        #     save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
        #                  , SAVE_DIR, 
        #                  ["actor", "q1", "q2", "target_q1", "target_q2"]
        #                  , episode
        #                  )
        #     save_tensor(alpha, SAVE_DIR, "alpha", episode)
        if episode % visulaize_step == 0:
            # if (len(total_steps) != 0) and (np.mean(step_done_set) < min(total_steps)):
            save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                        , SAVE_DIR, 
                        ["actor", "q1", "q2", "target_q1", "target_q2"]
                        , episode
                        )
            save_tensor(alpha, SAVE_DIR, "alpha", episode)
            total_steps = np.hstack((total_steps, np.mean(step_done_set)))
            save_numpy(total_steps, SAVE_DIR, "total_steps", episode)

            success = len(np.where(np.array(step_done_set) != MAX_STEP)[0])
            print("#{}: {}\t{:.2f}".format(episode, np.mean(step_done_set).astype(int), 100 * success / visulaize_step))
            live_plot(total_steps, visulaize_step)
            step_done_set = []

    # Turn the sim off
    save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                , SAVE_DIR, 
                ["actor", "q1", "q2", "target_q1", "target_q2"]
                , "save"
                )
    # Show the results
    show_result(total_steps, visulaize_step, SAVE_DIR)

else:
    del sim
    sim = DishSimulation(
                        # visualize=None,
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
    index = np.array(["dish", "try", "success", "failed"])
    idx = 0
    while True: 
        mode_change_step = 0
        # 0. Reset environment
        # state_curr, _, _,_ = sim.reset(mode="continous", slider_num=8)
        while True:
            if idx % 3 == 0:
                state_curr, _, _, mode = sim.reset(mode=None, slider_num=4)
            elif idx % 3 == 1:
                state_curr, _, _, mode = sim.reset(mode=None, slider_num=6)
            else:
                state_curr, _, _, mode = sim.reset(mode=None, slider_num=2)
            idx += 1
            state_curr1, state_curr2 = state_curr
            obs_num = np.sum(np.any(state_curr2, axis=1))

            if np.sum(dish[obs_num,0]) <= 150:
                break


        state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)

        # Running one episode
        for step in range(1, MAX_STEP + 1):
            # 1. Get action from policy network
            with torch.no_grad():
                action, logprob = actor_net.sample_action(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                action = action.squeeze().cpu().numpy()

            if step > 10:
                rand = (2 * np.random.random(action.size) - 1) * (step / MAX_STEP)
                rand[2:] *= 2
                action = np.clip(action + rand, -0.9999, 0.9999)

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done, mode = sim.env.step(action, mode)

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
        for step in range(1, MAX_STEP + 1):
            # 1. Get action from policy network
            with torch.no_grad():
                action, logprob = actor_net.sample_action(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))

            # if mode == 0:
            #     print(action.squeeze().cpu().numpy())

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
            print("reward", reward)
            if done: break

        dish[obs_num, 0] += 1
        if reward > 3: dish[obs_num, 1] += 1
        else:          dish[obs_num, 2] += 1


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
        time.sleep(1)

# Turn the sim off
sim.env.close()
