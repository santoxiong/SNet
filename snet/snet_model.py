import numpy as np
import torch
import os
from utils.dataset import MyDataset
from torch.utils.data import DataLoader, random_split


class SpectrumNet(torch.nn.Module):
    def __init__(self):
        super(SpectrumNet, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = torch.nn.Linear(1000, 500)
        self.hidden2 = torch.nn.Linear(500, 256)
        self.hidden3 = torch.nn.Linear(256, 128)

        self.branch1 = torch.nn.Linear(128, 64)
        self.output1 = torch.nn.Linear(64, 1)

        self.branch2 = torch.nn.Linear(128, 64)
        self.output2 = torch.nn.Linear(64, 1)

        self.branch3 = torch.nn.Linear(128, 64)
        self.output3 = torch.nn.Linear(64, 1)

        # Define sigmoid activation and softmax output
        self.active = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Share the representations
        x = self.active(self.hidden1(x))
        x = self.active(self.hidden2(x))
        x = self.active(self.hidden3(x))

        # Branch 1 -- classes
        x1 = self.active(self.branch1(x))
        x1 = self.active(self.output1(x1))
        a = self.softmax(x1)

        # Branch 2 -- sugar
        x2 = self.active(self.branch2(x))
        x2 = self.active(self.output2(x2))
        b = self.softmax(x2)

        # Branch 3 -- hardness
        x3 = self.active(self.branch3(x))
        x3 = self.active(self.output3(x3))
        c = self.softmax(x3)
        return a, b, c


