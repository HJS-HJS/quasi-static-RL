'''
PPO (Proximal Policy Optimization)
- continous action
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append(os.path.abspath("/home/rise/workspace/study/quasi-static-RL/src/third_party/quasi_static_push/scripts/"))
from third_party.quasi_static_push.scripts.dish_simulation import DishSimulation
from utils.policy_model_linear      import Network
from utils.utils             import live_plot, show_result, save_model, load_model
from collections             import namedtuple

## Parameters
TRAIN           = True
# Learning frame
FRAME = 8
# Learning Parameters
LEARNING_RATE   = 0.005   # optimizer
DISCOUNT_FACTOR = 0.99    # gamma
ADVANTAGE_LAMBDA= 0.95
EPISODES        = 20000   # total episode
# Memory
BATCH_SIZE = 128
EPOCH_SIZE = 10
CLIP_EPSILON = 0.05
# Other
visulaize_step = 5
MAX_STEP = 1024           # maximun available step per episode

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/model/single_target_linear"

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

# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 81
N_OUTPUT    = sim.env.action_space.shape[0] - 1  # 5

Transition = namedtuple('Transition',
                        ('state', 'action', 'prob', 'next_state', 'reward'))

# Memory
memory = []

# Network model
class ActorNetwork(Network):
    def __init__(self, n_state:int = N_INPUTS, n_action:int = N_OUTPUT, hidden_layer:int = 256):
        super(ActorNetwork, self).__init__(n_state, n_action, hidden_layer)
        self.mu = nn.Linear(128,n_action)
        self.std  = nn.Linear(128,n_action)

    def forward(self, state):
        x = self.layer(state)
        mu = 2 * torch.tanh(self.mu(x))
        # std = nn.functional.softplus(self.std(x))
        std = torch.exp(self.std(x))
        std = torch.clamp(std, min=0.2, max=2.0)
        dist = torch.distributions.MultivariateNormal(mu, torch.diag_embed(std, 0))
        return dist

class CriticNetwork(Network):
    def __init__(self, n_state:int = N_INPUTS, n_action:int = N_OUTPUT, hidden_layer:int = 256):
        super(CriticNetwork, self).__init__(n_state, n_action, hidden_layer)
        self.layer.add_module("linear", nn.Linear(128, 1))


# Initialize network
actor_net = ActorNetwork(N_INPUTS, N_OUTPUT, 256).to(device)
critic_net = CriticNetwork(N_INPUTS, N_OUTPUT, 256).to(device)

# Optimizer
actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
critic_optimizer = torch.optim.AdamW(critic_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

def optimize_model(batch):

    state_batch     = torch.cat(batch.state)
    action_batch    = torch.cat(batch.action)
    prob_old_batch  = torch.cat(batch.prob)
    next_state_batch= torch.cat(batch.next_state)
    reward_batch    = torch.cat(batch.reward)

    for i in range(EPOCH_SIZE):
        # Calculate the current probability
        distribution = actor_net(state_batch)
        prob_batch = distribution.log_prob(action_batch)

        # Calculate the importance ratio
        imp_ratio = torch.exp(prob_batch - prob_old_batch.detach())

        ## Calculate A from V
        # Calculate delta_t
        value_curr_set = critic_net(state_batch)
        value_next_set = critic_net(next_state_batch)
        td_target = reward_batch + DISCOUNT_FACTOR * value_next_set
        # delta_set = (td_target - value_curr_set).detach()
        advantage_set = (td_target - value_curr_set).detach()
        # Calculate Advantage_set
        # powers = (DISCOUNT_FACTOR * ADVANTAGE_LAMBDA) ** torch.arange(len(value_curr_set), dtype=torch.float32)
        # matrix = torch.tril(powers.view(-1, 1) / powers.view(1, -1)).to(device)
        # advantage_set = torch.mul(delta_set.repeat(1, len(value_curr_set)), matrix)
        # advantage_set = torch.sum(advantage_set, dim=0).unsqueeze(-1)

        # Clipping
        clip_advantage = torch.min(imp_ratio * advantage_set, torch.clamp(imp_ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantage_set)

        # Learning
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Policy Gradient 
        actor_loss = -clip_advantage.sum()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            actor_net.parameters(),
            0.5
        )
        actor_optimizer.step()

        criterion = torch.nn.MSELoss()
        critic_loss = criterion(td_target.detach(), value_curr_set).to(torch.float32)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            critic_net.parameters(),
            0.5
        )
        critic_optimizer.step()

total_steps = []
step_done_set = []
if TRAIN:
    for episode in range(1, EPISODES + 1):
        # if episode % 10 == 0:
        #     sim.env.close()
        #     sim = DishSimulation(
        #         visualize="human",
        #         state="linear",
        #         random_place=True,
        #         action_skip=FRAME,
        #         )

        # 0. Reset environment
        state_curr, _, _ = sim.env.reset(slider_num=0)
        state_curr = torch.tensor(state_curr.T, dtype=torch.float32, device=device)

        # Running one episode
        total_reward = 0.0
        for step in range(MAX_STEP):
            # 1. Get action from policy network
            distribution = actor_net(state_curr.unsqueeze(0))
            action = distribution.sample()

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done = sim.env.step(action[0].tolist())
            total_reward += reward

            # 3. Update state
            state_next = torch.tensor(state_next.T, dtype=torch.float32, device=device)
            # 4. Save data
            memory.append(Transition(
                state_curr.unsqueeze(0),
                action,
                distribution.log_prob(action),
                state_next.unsqueeze(0),
                torch.tensor([reward], device=device, dtype=torch.float32).unsqueeze(0),
            ))
            # 5. Update state
            state_curr = state_next

            # 6. Learning
            if (len(memory) % BATCH_SIZE == 0) or done:
                # print(len(memory))
                optimize_model(Transition(*zip(*memory)))
                memory = []
            if done: 
                # if reward > 5: print("grip success")
                # else: print("dish out of table")
                break

        ## Episode is finished
        print(episode, "\t reward ", total_reward, "\t step ", step)
        
        # Save episode reward
        step_done_set.append(total_reward)
        # Visualize
        if episode % visulaize_step == 0:
            if (len(total_steps) != 0) and (np.mean(step_done_set) >= max(total_steps)):
                save_model(actor_net, SAVE_DIR, "actor", episode)
                save_model(critic_net, SAVE_DIR, "critic", episode)
            total_steps.append(np.mean(step_done_set))
            print("#{}: {:.2f}".format(episode, np.mean(step_done_set)))
            live_plot(total_steps, visulaize_step)
            step_done_set = []

        # if episode % 10 == 0:
        #     sim.env.close()
        #     sim = DishSimulation(
        #         visualize=None,
        #         state="linear",
        #         random_place=True,
        #         action_skip=FRAME,
        #         )
    
    # Show the results
    show_result(total_steps, visulaize_step, SAVE_DIR)

else:
    sim = DishSimulation(visualize="Human",
                        state="linear",
                        random_place=True,
                        action_skip=5
                        )
    actor_net = load_model(actor_net, SAVE_DIR, "9325_actor")

    # 0. Reset environment
    state_curr, _ = sim.env.reset()
    state_curr = torch.tensor(state_curr.T, dtype=torch.float32, device=device)

    # Running one episode
    total_reward = 0.0
    for step in range(MAX_STEP):
        # 1. Get action from policy network
        distribution = actor_net(state_curr.unsqueeze(0))
        most_likely_action = distribution.mean

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done = sim.step(most_likely_action[0].tolist())
        state_curr = torch.tensor(state_next.T, dtype=torch.float32, device=device)
        total_reward += reward

# Turn the sim off
sim.env.close()
