'''
SAC (Soft Actor Critic)
- continous action
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import random
import time

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.abspath(current_directory + "/third_party/quasi_static_push/scripts/"))
from utils.dish_simulation6 import DishSimulation

from utils.sac_dataset2   import SACDataset
from utils.utils         import live_plot, show_result, save_models, save_tensor, load_model, load_models, load_tensor

## Parameters
# TRAIN           = False
TRAIN           = True
LOAD            = False
# LOAD            = True
FILE_NAME = None
# Learning frame
FRAME = 8
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
TARGET_UPDATE_TAU= 0.005
EPISODES        = 4000   # total episode
TARGET_ENTROPY  = -4.0
ALPHA           = 0.01
LEARNING_RATE_ALPHA= 0.01
# Memory
MEMORY_CAPACITY = 50000
BATCH_SIZE = 128
EPOCH_SIZE = 1
# Other
visulaize_step = 20
MAX_STEP = 250         # maximun available step per episode
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/../model/SAC_linear_6"

sim = DishSimulation(
    visualize=None,
    state="linear",
    random_place=True,
    action_skip=FRAME,
    )
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS1   = sim.env.observation_space[0].shape[0] # 81
N_INPUTS2   = sim.env.observation_space[1].shape[1] # 81
N_OUTPUT    = sim.env.action_space.shape[0]   # 5

# Memory
memory = SACDataset(MEMORY_CAPACITY)

class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.obs1_layer = nn.Sequential(
            nn.Conv1d(n_obs,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.t_net1 = nn.Sequential(
            nn.Conv1d(64,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.t_net2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 64*64),
            nn.ReLU(),
        )
            
        self.obs2_layer = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.obs3_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(256,n_action),
        )
        self.std = nn.Sequential(
            nn.Linear(256,n_action),
            nn.Softplus(),
        )

    def forward(self, state, obs):
        x = self.layer(state)
        obs = self.obs1_layer(obs)
        _t = self.t_net1(obs)
        _t = torch.max(_t, 2, keepdim=True)[0]
        _t = self.t_net2(_t)
        _t = _t.view(-1, 64, 64)
        obs = torch.bmm(_t, obs)
        obs = self.obs2_layer(obs)
        obs = torch.max(obs, 2, keepdim=True)[0]
        obs = self.obs3_layer(obs)

        x=self.last_layer(torch.cat([x, obs], dim=1))

        mu = self.mu(x)
        std = self.std(x)

        # sample
        distribution = torch.distributions.Normal(mu, std)
        u = distribution.rsample()
        logprob = distribution.log_prob(u)

        # Enforce action bounds [-1., 1.]
        action = torch.tanh(u)
        logprob = logprob - torch.log(1 - torch.tanh(u).pow(2) + 1e-7)

        return action, logprob

class QNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_obs:int = 4, n_action:int = 2):
        super(QNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(n_state, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.obs1_layer = nn.Sequential(
            nn.Conv1d(n_obs,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.t_net1 = nn.Sequential(
            nn.Conv1d(64,64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.t_net2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 64*64),
            nn.ReLU(),
        )   
        self.obs2_layer = nn.Sequential(
            nn.Conv1d(64,128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.obs3_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_layer = nn.Sequential(
            nn.Linear(n_action, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, obs, action):
        _state = self.state_layer(state)
        _obs = self.obs1_layer(obs)
        _t = self.t_net1(_obs)
        _t = torch.max(_t, 2, keepdim=True)[0]
        _t = self.t_net2(_t)
        _t = _t.view(-1, 64, 64)
        _obs = torch.bmm(_t, _obs)
        _obs = self.obs2_layer(_obs)
        _obs = torch.max(_obs, 2, keepdim=True)[0]
        _obs = self.obs3_layer(_obs)
        _action = self.action_layer(action)
        return self.layer(torch.cat([_state, _action, _obs], dim=1))
    
    def train(self, target, state, obs, action, optimizer):
        criterion = torch.nn.SmoothL1Loss()
        optimizer.zero_grad()
        loss = criterion(self.forward(state, obs, action) , target)
        loss.mean().backward()
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

if LOAD:
    actor_net, q1_net, q2_net, target_q1_net, target_q2_net = load_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                    , SAVE_DIR, 
                    ["actor", "q1", "q2", "target_q1", "target_q2"]
                    , FILE_NAME
                    )
    alpha = load_tensor(alpha, SAVE_DIR, "alpha", FILE_NAME)
    alpha.requires_grad = True

# Optimizer
actor_optimizer   = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE)
q1_optimizer = torch.optim.AdamW(q1_net.parameters(), lr=LEARNING_RATE)
q2_optimizer = torch.optim.AdamW(q2_net.parameters(), lr=LEARNING_RATE)
alpha_optimizer   = torch.optim.AdamW([alpha], lr=LEARNING_RATE_ALPHA)

def optimize_model(batch):

    s1, s2, a, r, next_s1, next_s2 = batch

    # Calculate 
    with torch.no_grad():
        next_a, next_logprob = actor_net(next_s1, next_s2)
        next_entropy = -alpha.exp() * next_logprob.mean(dim=1)

        next_q1 = target_q1_net(next_s1, next_s2, next_a)
        next_q2 = target_q2_net(next_s1, next_s2, next_a)

        next_min_q = torch.min(torch.cat([next_q1, next_q2], dim=1), 1)[0]
        target = r + DISCOUNT_FACTOR * (next_min_q + next_entropy).unsqueeze(1)

    q1_net.train(target, s1, s2, a, q1_optimizer)
    q2_net.train(target, s1, s2, a, q2_optimizer)

    action, logprob_batch = actor_net(s1, s2)
    entropy = -alpha.exp() * logprob_batch.mean(dim=1)
    q1 = q1_net(s1, s2, action)
    q2 = q2_net(s1, s2, action)
    
    min_q = torch.min(torch.cat([q1, q2],dim=1), 1)[0]
    actor_loss = -entropy - min_q
    actor_optimizer.zero_grad()
    actor_loss.mean().backward()
    actor_optimizer.step()

    alpha_optimizer.zero_grad()
    alpha_loss = - alpha.exp() * (logprob_batch.detach() + TARGET_ENTROPY).mean()
    alpha_loss.backward()
    alpha_optimizer.step()

    q1_net.update(target_q1_net)
    q2_net.update(target_q2_net)

total_steps = []
step_done_set = []
if TRAIN:
    for episode in range(1, EPISODES + 1):

        # 0. Reset environment
        if (episode // 2) % 4 == 0:
            state_curr, _, _ = sim.reset(mode=None, slider_num=2)
        elif (episode // 2) % 4 == 1:
            state_curr, _, _ = sim.reset(mode=None, slider_num=6)
        elif (episode // 2) % 4 == 2:
            state_curr, _, _ = sim.reset(mode="continous", slider_num=5)
        elif (episode // 2) % 4 == 3:
            state_curr, _, _ = sim.reset(mode="pusher", slider_num=8)

        state_curr1 = torch.tensor(state_curr[0], dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr[1].T, dtype=torch.float32, device=device)

        # Running one episode
        total_reward = 0.0
        start_time = time.time()
        for step in range(1, MAX_STEP + 1):
            # 1. Get action from policy network
            action, logprob = actor_net(state_curr1, state_curr2.unsqueeze(0))

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done = sim.env.step(action[0].tolist())
            total_reward += reward

            # 3. Update state
            state_next1 = torch.tensor(state_next[0], dtype=torch.float32, device=device).unsqueeze(0)
            state_next2 = torch.tensor(state_next[1].T, dtype=torch.float32, device=device)

            # 4. Save data
            memory.push(
                state_curr1,
                state_curr2.T,
                action,
                torch.tensor([reward], device=device).unsqueeze(0),
                state_next1,
                state_next2.T,
            )

            # 5. Update state
            state_curr1, state_curr2 = state_next1, state_next2

            # 6. Learning
            if (len(memory) > BATCH_SIZE):
                for _ in range(EPOCH_SIZE):
                    optimize_model(memory.sample(BATCH_SIZE))

            if done: 
                break

        ## Episode is finished
        print("\t", episode, "\t", step, "\t{:.2f}\t{:.2f}".format(total_reward, time.time() - start_time))
        if done and (reward < 5): step = MAX_STEP
        
        # Save episode reward
        step_done_set.append(step)
        # Visualize
        if (len(total_steps) != 0) and (step < min(total_steps)):
            save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                         , SAVE_DIR, 
                         ["actor", "q1", "q2", "target_q1", "target_q2"]
                         , episode
                         )
            save_tensor(alpha, SAVE_DIR, "alpha", episode)
        if episode % visulaize_step == 0:
            if (len(total_steps) != 0) and (np.mean(step_done_set) < min(total_steps)):
                save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                            , SAVE_DIR, 
                            ["actor", "q1", "q2", "target_q1", "target_q2"]
                            , episode
                            )
                save_tensor(alpha, SAVE_DIR, "alpha", episode)
            total_steps.append(np.mean(step_done_set))
            print("#{}: ".format(episode), np.mean(step_done_set).astype(int))
            live_plot(total_steps, visulaize_step)
            step_done_set = []

    # Turn the sim off
    sim.env.close()
    save_models([actor_net, q1_net, q2_net, target_q1_net, target_q2_net]
                , SAVE_DIR, 
                ["actor", "q1", "q2", "target_q1", "target_q2"]
                , "save"
                )
    # Show the results
    show_result(total_steps, visulaize_step, SAVE_DIR)

else:
    del sim
    sim = DishSimulation(visualize="human",
                        state="linear",
                        random_place=True,
                        action_skip=FRAME
                        )
    actor_net = load_model(actor_net, SAVE_DIR, "actor", FILE_NAME)
    q1_net = load_model(q1_net, SAVE_DIR, "q1", FILE_NAME)
    q2_net = load_model(q2_net, SAVE_DIR, "q2", FILE_NAME)
    target_q1_net = load_model(target_q1_net, SAVE_DIR, "target_q1", FILE_NAME)
    target_q2_net = load_model(target_q2_net, SAVE_DIR, "target_q2", FILE_NAME)
    alpha = load_tensor(alpha, SAVE_DIR, "alpha", FILE_NAME)

    while True: 
        # 0. Reset environment
        state_curr, _, _ = sim.reset(mode="continous", slider_num=8)
        state_curr1 = torch.tensor(state_curr[0], dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr[1].T, dtype=torch.float32, device=device)

        # Running one episode
        for step in range(MAX_STEP):
            # 1. Get action from policy network
            action, logprob = actor_net(state_curr1, state_curr2.unsqueeze(0))

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done = sim.env.step(action[0].tolist())
            state_curr1 = torch.tensor(state_next[0], dtype=torch.float32, device=device).unsqueeze(0)
            state_curr2 = torch.tensor(state_next[1].T, dtype=torch.float32, device=device)
            if done: break
        print("Episode finished with step #{}".format(step))

# Turn the sim off
sim.env.close()
