import os
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import torch
from typing import List

def plot(total_steps, step):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.linspace(1, len(total_steps) + 1, len(total_steps), endpoint=False) * step, total_steps, c='red')
    plt.grid(True)

def plots(total_steps_set, step):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    # y축 간격 10으로 설정
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(10))

    color = ['red', 'blue', 'black', 'green', 'gray']
    for idx, total_steps in enumerate(total_steps_set):
        plt.plot(np.linspace(1, len(total_steps) + 1, len(total_steps), endpoint=False) * step, total_steps, c=color[idx])
    plt.grid(True)

def live_plot(total_steps, step):
    plot(total_steps, step)
    plt.pause(0.0001)

def live_plots(total_steps_set, step):
    plots(total_steps_set, step)
    plt.pause(0.0001)

def show_result(total_steps, step, save_dir:None):
    print('step mean:', np.mean(total_steps))
    print('step  std:', np.std(total_steps))
    print('step  min:', np.min(total_steps))
    print('step  max:', np.max(total_steps))
    plot(total_steps, step)
    if save_dir is not None:
        plt.savefig(save_dir + "/results.png")
    plt.show()

def show_results(total_steps_set, step, save_dir:None):
    print('step mean:', np.mean(total_steps_set[0]))
    print('step  std:', np.std(total_steps_set[0]))
    print('step  min:', np.min(total_steps_set[0]))
    print('step  max:', np.max(total_steps_set[0]))
    plots(total_steps_set, step)
    if save_dir is not None:
        plt.savefig(save_dir + "/results.png")
    plt.show()

def save_model(network:torch.nn.Module, save_dir:str, name:str, episode:int):
    _save_dir = save_dir + "/" + str(episode) + "/"
    os.makedirs(_save_dir, exist_ok=True)
    save_path = os.path.join(_save_dir, f"{name}.pth")
    torch.save(network.state_dict(), save_path)
    print(f"\t\tModels saved at episode {episode}")

def save_models(networks:List[torch.nn.Module], save_dir:str, names:List[str], episode:int):
    _save_dir = save_dir + "/" + str(episode) + "/"
    os.makedirs(_save_dir, exist_ok=True)
    for net, name in zip(networks, names):
        save_path = os.path.join(_save_dir, f"{str(name)}.pth")
        torch.save(net.state_dict(), save_path)
    print(f"\t\tModels saved at episode {episode}")

def save_tensor(tensor:torch.tensor, save_dir:str, name:str, episode:int):
    _save_dir = save_dir + "/" + str(episode) + "/"
    os.makedirs(_save_dir, exist_ok=True)
    save_path = os.path.join(_save_dir, f"{str(name)}.pth")
    torch.save(tensor, save_path)
    # print(f"\t\tModels saved at episode {episode}")

def save_numpy(numpy:np.array, save_dir:str, name:str, episode:int):
    _save_dir = save_dir + "/" + str(episode) + "/"
    os.makedirs(_save_dir, exist_ok=True)
    save_path = os.path.join(_save_dir, f"{str(name)}")
    np.save(save_path, numpy)
    # print(f"\t\tModels saved at episode {episode}")

def load_model(network:torch.nn.Module, save_dir:str, name:str, episode:int):
    if episode is None:
        folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
        episode = max(numbered_folders)

    _save_dir = save_dir + "/" + str(episode) + "/"
    save_path = os.path.join(_save_dir, f"{str(name)}.pth")
    if os.path.exists(_save_dir):
        state_dict = torch.load(save_path, weights_only=True)
        network.load_state_dict(state_dict)
        print(f"\tModel loaded successfully {episode}")
    else:
        print(f"\tNo saved model found {save_path}")
    return network

def load_models(networks:List[torch.nn.Module], save_dir:str, names:List[str], episode:int):
    if episode is None:
        folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
        episode = max(numbered_folders)

    _save_dir = save_dir + "/" + str(episode) + "/"
    for idx in range(len(networks)):
        save_path = os.path.join(_save_dir, f"{str(names[idx])}.pth")
        if os.path.exists(_save_dir):
            state_dict = torch.load(save_path, weights_only=True)
            networks[idx].load_state_dict(state_dict)
        else:
            print(f"\tNo saved model found {save_path}")
    print("\tModels loaded successfully")
    return networks

def load_tensor(tensor:torch.tensor, save_dir:str, name:str, episode:int):
    if episode is None:
        folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
        episode = max(numbered_folders)
        
    _save_dir = save_dir + "/" + str(episode) + "/"
    save_path = os.path.join(_save_dir, f"{str(name)}.pth")
    if os.path.exists(save_path):
        tensor = torch.load(save_path, weights_only=False)
        print("\tTensor loaded successfull")
    else:
        print(f"\tNo saved model found {save_path}")
    return tensor

def load_episode(save_dir:str, name:str):
    folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
    numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
    episode = max(numbered_folders)
    return episode

def load_numpy(numpy:np.array, save_dir:str, name:str, episode:int):
    if episode is None:
        folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
        episode = max(numbered_folders)
        
    _save_dir = save_dir + "/" + str(episode) + "/"
    save_path = os.path.join(_save_dir, f"{str(name)}.npy")
    if os.path.exists(save_path):
        numpy = np.load(save_path)
        print("\tNumpy loaded successfull")
    else:
        print(f"\tNo saved numpy found {save_path}")
    return np.array(numpy)

def visualize_attention(attn_weights, title="Attention Map"):
    """
    어텐션 가중치를 시각화하는 함수
    Args:
        attn_weights: [batch_size, num_heads, seq_length, seq_length] 크기의 텐서
    """
    attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()  # 배치 제거
    num_heads = attn_weights.shape[0]

    fig, axs = plt.subplots(1, num_heads, figsize=(15, 5))
    axs = np.atleast_1d(axs)  # axs를 항상 배열로 처리

    fig.suptitle(title)

    for i in range(num_heads):
        axs[i].imshow(attn_weights[i], cmap='viridis')
        axs[i].set_title(f"Head {i+1}")
        axs[i].axis('off')

    plt.show()

def last_file(save_dir:str):
    folders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
    numbered_folders = [int(folder) for folder in folders if folder.isdigit()]
    return max(numbered_folders)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model(actor_net, q1_net, q2_net, test_dish, device, vis, record, sim):
    max_check_num = 500
    dish = np.zeros((10, 3), dtype=int)
    fail_case = np.zeros((10, 3), dtype=int)
    index = np.array(["dish", "try", "success", "failed", "rate", "g fail", "d out", "s out"])
    target_distance = 0
    target_distance_set = np.zeros(10, dtype=float)
    while True: 
        mode_change_step = 0
        while True:
            # if idx % 3 == 0:
            #     _num = test_dish + 1
            # elif idx % 3 == 1:
            #     _num = test_dish - 3
            # else:
            #     _num = test_dish // 2
            # state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num)
            # limit = _num - 3
            # state_curr1, state_curr2 = state_curr
            # obs_num = np.sum(np.any(state_curr2, axis=1))

            # if obs_num < limit: continue
            # idx += 1
            # obs_num -= 1
            # if np.sum(dish[obs_num,0]) < 200:
            #     break
            valid_indices = np.where(dish[:test_dish, 0] < max_check_num)[0]    # max_check_num 미만인 값들의 인덱스

            if len(valid_indices) > 0:
                # _num = np.random.choice(valid_indices)
                _num = valid_indices[-1]
                state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num + 1)
                # state_curr, _, _, mode = sim.reset(mode="continuous", slider_num=_num + 1)
                state_curr1, state_curr2 = state_curr
                obs_num = np.sum(np.any(state_curr2, axis=1)) - 1
                if obs_num < (_num - 3): 
                    state_curr, _, _, mode = sim.reset(mode=None, slider_num=_num + 1)
                    state_curr1, state_curr2 = state_curr
                    obs_num = np.sum(np.any(state_curr2, axis=1)) - 1
                if dish[obs_num, 0] < max_check_num: break
            else:
                print("finished")
                return

        target_init_pose = np.array(sim.env.get_state()["slider_state"][0])

        state_curr1 = torch.tensor(state_curr1, dtype=torch.float32, device=device).unsqueeze(0)
        state_curr2 = torch.tensor(state_curr2.T, dtype=torch.float32, device=device)

        # Running one episode
        for step in range(1, 200 + 1):
            # 1. Get action from policy network
            with torch.no_grad():
                action, std = actor_net(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                _action = action.squeeze().cpu().numpy()
                _action = np.tanh(_action)

            if step > 10:
                rand = (2 * np.random.random(_action.size) - 1) * (step / 200)
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

        # for step in range(1, 200 + 1):
        #     rand = np.random.random(4)
        #     rand_summon_action = (rand / 2 + 0.5 * np.sign(rand)).astype(np.float32)
        #     state_next, reward, done, mode_next = sim.env.step(rand_summon_action, mode)
        #     if mode_next == 1: break
        # if mode_next == 0: continue
        # state_next1, state_next2 = state_next
        # state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
        # state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
        # state_curr1 = state_next1
        # state_curr2 = state_next2
        
        # Running one episode
        if not done: 
            for step in range(1, 200 + 1):
                # 1. Get action from policy network
                target_distance = np.linalg.norm(target_init_pose - np.array(sim.env.get_state()["slider_state"][0]))
                with torch.no_grad():
                    action, std = actor_net(state_curr1, state_curr2.unsqueeze(0), torch.tensor([mode], device=device).unsqueeze(0))
                    # action, std = actor_net(state_curr1, state_curr2.unsqueeze(0))
                _action = action.squeeze().cpu().numpy()
                _action = np.tanh(_action)
                # _action[3] = 1
                # 2. Run simulation 1 step (Execute action and observe reward)
                state_next, reward, done, mode = sim.env.step(_action, mode)

                if mode == 0:
                    mode_change_step = step
                else :
                    state_next1, state_next2 = state_next
                    state_next1 = torch.tensor(state_next1, dtype=torch.float32, device=device).unsqueeze(0)
                    state_next2 = torch.tensor(state_next2.T, dtype=torch.float32, device=device)
                    state_curr1 = state_next1
                    state_curr2 = state_next2
                # print("reward", reward)
                # print("action", _action)
                # print("std", std.cpu().numpy())
                # print("q1", q1_net(state_next1, state_next2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))
                # print("q2", q2_net(state_next1, state_next2.unsqueeze(0), action, torch.tensor([mode], device=device).unsqueeze(0)))
                if done: break

        dish[obs_num, 0] += 1
        if reward > 0.5: dish[obs_num, 1] += 1
        else:            dish[obs_num, 2] += 1
        target_distance_set[obs_num] += target_distance

        if vis: time.sleep(1)

        if reward < 0:
            if   reward == -2.0: 
                fail_case[obs_num, 0] += 1
                if record: sim.env.simulator.renameSavedFiles("", "g_fail", True)
            elif reward < -2.0: 
                fail_case[obs_num, 1] += 1
                if record: sim.env.simulator.renameSavedFiles("", "d_out", True)
            else: 
                fail_case[obs_num, 2] += 1
                if record: sim.env.simulator.renameSavedFiles("", "s_out", True)
        print("\tEpisode finished")
        print("\t\tStep #{}\tMode changed #{}".format(step, mode_change_step))
        print("\t\tSuccess Rate {:.2f}%\tTry {}\tSuccess {}\tFail {}".format(np.sum(dish[:,1]) / np.sum(dish[:,0]) * 100, np.sum(dish[:,0]), np.sum(dish[:,1]), np.sum(dish[:,2])))
        print("\t\t{}".format(index[0]), end="\t")
        for i in range(10):
            print("#{}".format(i), end="\t")
        print("")
        for j in range(3):
            print("\t\t{}".format(index[j + 1]), end="\t")
            for i in range(10):
                print("{}".format(dish[i, j]), end="\t")
            print("")

        # success rate
        print("\t\t{}".format(index[4]), end="\t")
        for i in range(10):
            print("{:.1f}".format(dish[i, 1] / (dish[i, 0] + 1e-6) * 100), end="\t")
        print("")

        # fail case
        for j in range(3):
            print("\t\t{}".format(index[j + 5]), end="\t")
            for i in range(10):
                print("{}".format(fail_case[i, j]), end="\t")
            print("{}".format(np.sum(fail_case[:,j])), end="\t")
            print("")

        # moving distance
        print("\t\tmv dist", end="\t")
        for i in range(10):
            print("{:.3f}".format(np.sum(target_distance_set[i]) / (dish[i, 0] + 1e-6)), end="\t")
        print("")
        if vis: time.sleep(0.5)
    return