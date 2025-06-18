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

so_file_path = os.path.abspath("../simulator")
sys.path.append(so_file_path)

from utils.simulation_wrapping import DishSimulation

from utils.sac_dataset import SACDataset
from utils.utils           import *

## Parameters
TRAIN           = False
# TRAIN           = True
curriculum = 5

# FILE_NAME = None
FILE_NAME = 12000

loss = 0.

# Learning frame
FRAME = 20
# Learning Parameters
LEARNING_RATE   = 0.0004 # optimizer
DISCOUNT_FACTOR = 0.97   # gamma
TARGET_UPDATE_TAU= 0.01
EPISODES        = 15000   # total episode
ALPHA           = 0.5
LEARNING_RATE_ALPHA= 0.0005 # Memory
MEMORY_CAPACITY = 80000
BATCH_SIZE = 256
EPOCH_SIZE = 2
# Other
visulaize_step = 50
MAX_STEP = 150         # maximun available step per episode
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/../model/test_mean_max"
episode_start = 1

# curriculum
curriculum_dictionary = np.array([
    # obs, action_step, target_entropy
    [3, 3, -1],
    [4, 4, -1],
    [5, 4, -2],
    [6, 4, -2],
    [8, 4, -3],
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
N_INPUTS1   = 15
N_INPUTS2   = 16
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
    def __init__(self, obs_dim=10, hidden_dim=1024):
        super(SelfAttentionObstacle, self).__init__()
        hidden_hiddden_dim = int(hidden_dim / 2)
        self.mean_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_hiddden_dim),
            nn.ReLU(),
            nn.Linear(hidden_hiddden_dim, hidden_hiddden_dim),
            nn.ReLU(),
        )
        self.max_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_hiddden_dim),
            nn.ReLU(),
            nn.Linear(hidden_hiddden_dim, hidden_hiddden_dim),
            nn.ReLU(),
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
            nn.Linear(n_state, 512),
            nn.ReLU(),
        )

        self.self_attention = nn.ModuleList([
            SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=512)
            for _ in range(2)
            ])

        self.mu = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
            ) for _ in range(2)
        ])

        self.std = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_action),
                nn.Softplus(),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, mode):
        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)
        # State branch
        _state = self.layer(state)  # [batch, 1024]


        if mode.numel() == 1:
            mode_idx = mode.item()
            # Obs self attention
            _obs = self.self_attention[mode_idx](obs)  # [batch, 1024]
            _state = torch.cat([_state, _obs], dim=1)
            mu = self.mu[mode_idx](_state)
            std = self.std[mode_idx](_state) + 1e-3
        else:
            # Obs self attention
            _obs = torch.where(mode.bool(), self.self_attention[1](obs), self.self_attention[0](obs))
            _state = torch.cat([_state, _obs], dim=1)
            mu = torch.where(mode.bool(), self.mu[1](_state), self.mu[0](_state))
            std = torch.where(mode.bool(), self.std[1](_state), self.std[0](_state)) + 1e-3

        return mu, torch.clamp(std, min=0.1, max=2.0)
    

    def sample_action(self, state, obs, mode):
        mu, std = self.forward(state, obs, mode)

        _cut = curriculum_dictionary[curriculum][1]
        use_idx = {
            2: [0, 1],             # x, y
            3: [0, 1, 2],          # x, y, theta
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
                theta = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(-0.3, 0.3).to(action_countinue.device)  # fixed theta
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 0.8).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, theta, width], dim=1)
            
            elif _cut == 3:
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 0.8).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, width], dim=1)

            else:
                pass
        else:
            if _cut == 2:
                theta = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(-0.3, 0.3).to(action_countinue.device)
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 0.8).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, theta, width], dim=1)
            
            elif _cut == 3:
                width = torch.FloatTensor(action_countinue.shape[0], 1).uniform_(0.5, 0.8).to(action_countinue.device)
                action_countinue = torch.cat([action_countinue, width], dim=1)

            else:
                pass
        return action_countinue, log_prob_continue.sum(dim = 1)


class QNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(QNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
        )

        self.action_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_action, 256),
                nn.ReLU(),
            ) for _ in range(2)
        ])

        self.self_attention = nn.ModuleList([
            SelfAttentionObstacle(obs_dim=n_obs, hidden_dim=512)
            for _ in range(2)
            ])

        self.layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + 512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ) for _ in range(2)
        ])

    def forward(self, state, obs, action, mode):
        mode = mode.long().view(-1, 1)
        mode_onehot = torch.zeros(state.size(0), 2, device=state.device)
        mode_onehot.scatter_(1, mode, 1.0)

        # Run
        _state = self.state_layer(state)  # [batch, 1024]
        _action = torch.where(mode.bool(), self.action_layer[0](action), self.action_layer[1](action))  # [batch, 1024]

        # Obs self attention
        _obs = torch.where(mode.bool(), self.self_attention[1](obs), self.self_attention[0](obs))
        fusion_input = torch.cat([_state, _obs, _action], dim=1)  # [batch, 768]

        _q = torch.where(mode.bool(), self.layer[1](fusion_input), self.layer[0](fusion_input))

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

print("전체 파라미터 수:", count_parameters(actor_net))

step_done_set = []
prev_curriculum_episode = episode_start
obs_num = 0
if TRAIN:
    for episode in range(episode_start, EPISODES + 1):
        limit = 0
        if (episode % 4) == 0:   
            state_curr, _, _, mode = sim.reset(mode=None, slider_num=curriculum_dictionary[curriculum][0])

        if (episode // 4) % 4 == 0:   
            _num = curriculum_dictionary[curriculum][0]
            state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num)
        elif (episode // 4) % 4 == 1: 
            _num = curriculum_dictionary[curriculum][0]
            state_curr, _, _, mode = sim.reset(mode="continuous", slider_num=_num)
        elif (episode // 4) % 4 == 2: 
            _num2 = max(3, curriculum_dictionary[curriculum][0] - 2)
            _num1 = _num2 - 2
            _num = np.random.randint(_num1, _num2)
            state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num)
        elif (episode // 4) % 4 == 3:
            _num1 = max(3, curriculum_dictionary[curriculum][0] - 2)
            _num = np.random.randint(2, _num1)
            state_curr, _, _, mode = sim.reset(mode="continuous", slider_num=_num)

        state_curr1, state_curr2 = state_curr
        valid_counts = np.sum(np.abs(state_curr2), axis = 1)  # [batch]
        obs_num = len(valid_counts) - len(np.where(valid_counts == 0)[0])

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

            # if done != 0: _done = 1
            # else: _done = 0
            if mode_next == 0:
                mode_change_step = step
            #     _done = 1
                
            # 3. Update state
            state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
            state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)

            # 4. Save data
            memory.push(
                # state_curr1.to(device=torch.device('cpu')),
                state_curr1.to(device=device),
                state_curr2.T,
                torch.tensor([mode], device=device).unsqueeze(0),
                torch.tensor(action, device=device).unsqueeze(0),
                torch.tensor([reward], device=device).unsqueeze(0),
                torch.tensor([done], device=device).unsqueeze(0),
                # state_next1.to(device=torch.device('cpu')),
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
    VIS = False
    RECORD = False
    del sim
    sim = DishSimulation(
                        visualize="human" if VIS else None,
                        state="linear",
                        record=RECORD,
                        save_dir = os.path.abspath("../recording"),
                        action_skip=FRAME
                        )
    actor_net = load_model(actor_net, SAVE_DIR, "actor", FILE_NAME)
    q1_net = load_model(q1_net, SAVE_DIR, "q1", FILE_NAME)
    q2_net = load_model(q2_net, SAVE_DIR, "q2", FILE_NAME)
    target_q1_net = load_model(target_q1_net, SAVE_DIR, "target_q1", FILE_NAME)
    target_q2_net = load_model(target_q2_net, SAVE_DIR, "target_q2", FILE_NAME)
    alpha = load_tensor(alpha, SAVE_DIR, "alpha", FILE_NAME)

    test_model(
        actor_net=actor_net,
        q1_net=q1_net,
        q2_net=q2_net,
        test_dish=curriculum_dictionary[curriculum][0],
        # test_dish=1,
        device=device,
        vis=VIS,
        record = RECORD,
        sim=sim
        )
# 12000
# Success Rate 89.64%	Try 4500	Success 4034	Fail 466
# dish	    #0	#1	#2	#3	#4	#5	#6	#7	#8	#9	
# try	    500	500	500	500	500	500	500	500	500	0	
# success	499	491	475	445	447	437	415	417	408	0	
# failed	1	9	25	55	53	63	85	83	92	0	
# rate	    998	982	950	890	894	874	830	834	816	00	
# g fail	0	2	5	10	13	10	19	27	26	0	112	
# d out	    0	0	8	26	21	22	23	28	26	0	154	
# s out	    1	7	12	19	19	31	43	28	40	0	200	
# mv dist	0.007	0.005	0.015	0.022	0.025	0.033	0.034	0.042	0.044	0.000	