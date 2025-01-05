import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, in_channel:int = 3, out_channel: int = 3):
        
        super(Network, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 100),
            nn.ReLU(),
        )
        
    def forward(self, state):
        return self.layer(state)

if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device('cuda')

    model = Network().eval()
    print(list(model.children()))
