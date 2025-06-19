import torch
import torch.nn as nn


class AdversarialDiscriminator(nn.Module):
    def __init__(self, volume_size):
        super(AdversarialDiscriminator, self).__init__()
        self.volume_size = volume_size

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * (volume_size[0] // 8) * (volume_size[1] // 8) * (volume_size[2] // 8), 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, volume):
        x = self.relu(self.conv1(volume))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x