import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=13, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + res
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        self.model_type = 'ResNet'
        in_channels = config['ts_dim']
        num_classes = config['num_classes']

        # 标准时序 ResNet 结构：包含 3 个通道递增的残差块
        self.block1 = ResNetBlock(in_channels, 32)
        self.block2 = ResNetBlock(32, 64)
        self.block3 = ResNetBlock(64, 128)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x