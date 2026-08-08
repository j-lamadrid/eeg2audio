import numpy as np
import os

import scipy.io
from scipy import signal

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import torch.optim as optim

import utils
import random
from diffusers import DiffusionPipeline


class EEG2Mel(nn.Module):

    def __init__(self):
        super(EEG2Mel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)

        # Batch Normalization Layers
        self.bn2d_1 = nn.BatchNorm2d(8)
        self.bn2d_2 = nn.BatchNorm2d(16)
        self.bn2d_3 = nn.BatchNorm2d(32)
        self.bn2d_4 = nn.BatchNorm2d(64)
        self.bn2d_5 = nn.BatchNorm2d(128)
        self.bn2d_6 = nn.BatchNorm2d(256)

        # Pooling Layer
        self.pool = nn.MaxPool2d(2)

        self.zero_padding = nn.ZeroPad2d(1)

        # Dropout Layers
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.15)

        # Fully Connected Layers
        self.fc1 = nn.Linear(119040, 201)  # Adjust this if the input size changes
        self.fc2 = nn.Linear(201, 201)
        self.linear = nn.Linear(201, 201 * 221)

        # Batch Normalization for Fully Connected Layers
        self.bn1d_1 = nn.BatchNorm1d(201)
        self.bn1d_2 = nn.BatchNorm1d(201)
        self.bn1d_3 = nn.BatchNorm1d(201 * 221)

        # Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.bn2d_1(F.relu(self.conv1(x))))
        x = self.dropout1(self.bn2d_2(F.relu(self.conv2(x))))
        x = self.dropout1(self.bn2d_3(F.relu(self.conv3(x))))
        x = self.dropout1(self.bn2d_4(F.relu(self.conv4(x))))
        x = self.dropout1(self.bn2d_5(F.relu(self.conv5(x))))
        x = self.zero_padding(F.relu(self.conv6(x)))  # Apply padding before batch norm and pooling
        x = self.pool(self.bn2d_6(x))

        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.dropout2(self.bn1d_1(F.relu(self.fc1(x))))
        x = F.relu(self.bn1d_2(self.fc2(x)))
        x = self.linear(x)
        x = self.bn1d_3(x)
        x = x.view(-1, 201, 221)

        return x


class EEGProjector(nn.Module):

    def __init__(self, Fy, Sy, Fz, Sz, Dz):
        super(EEGProjector, self).__init__()
        self.Fy = Fy  # eeg channels
        self.Sy = Sy  # time steps
        self.Fz = Fz  # latent freq
        self.Sz = Sz  # latent time
        self.Dz = Dz  # latent channels

        self.proj_layers = nn.Sequential(
            nn.Conv1d(Fy, 256, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(1024, 2048, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Linear(2048 * 6, Fz * Sz * Dz)

    def forward(self, eeg):
        eeg_projected = self.proj_layers(eeg)
        batch_size, reduced_Fy, reduced_Sy = eeg_projected.shape
        eeg_projected = eeg_projected.view(batch_size, -1)
        eeg_projected = self.fc(eeg_projected)
        eeg_projected = eeg_projected.view(batch_size, self.Fz, self.Sz, self.Dz)
        return eeg_projected


class ControlNet(nn.Module):
    def __init__(self, Fy, Sy, Fz, Sz, Dz):
        super(ControlNet, self).__init__()
        pipeline = DiffusionPipeline.from_pretrained("cvssp/audioldm2-music")
        pipeline = pipeline.to(device)
        self.base_model = pipeline
        self.unet = base_model.unet
        self.projector = EEGProjector(Fy, Sy, Fz, Sz, Dz)
        self.zero_conv = nn.Conv2d(Fz, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, z, eeg, t):
        eeg_projected = self.projector(eeg)
        condition = self.zero_conv(eeg_projected) + self.zero_conv(z)
        condition = condition.squeeze(1)
        pred = self.unet(z, t,
                         encoder_hidden_states=condition)
        return pred