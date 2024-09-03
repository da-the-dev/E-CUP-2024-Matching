import torch.nn as nn
import torch.nn.functional as F


class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward_one(self, x):
        return self.fc(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
