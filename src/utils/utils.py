import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List

def plot(total_steps, step):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.linspace(1, len(total_steps) + 1, len(total_steps), endpoint=False) * step, total_steps, c='red')
    plt.grid(True)

def live_plot(total_steps, step):
    plot(total_steps, step)
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
        print("\tModel loaded successfully")
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
