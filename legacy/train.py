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


def train(model,
          lr=0.0015,
          n_epochs=1000,
          batch_size=10,
          output_path='eeg2audio.pth'):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_epochs = n_epochs
    batch_size = batch_size

    combined = list(zip(eeg_X, spec_y))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        random.shuffle(combined)
        eeg_X_shuffled, spec_y_shuffled = zip(*combined)

        batch_eeg_tensors = []
        batch_spec_tensors = []

        for i in range(len(eeg_X_shuffled)):
            batch_eeg_tensors.append(eeg_X_shuffled[i])
            batch_spec_tensors.append(spec_y_shuffled[i])

            if len(batch_eeg_tensors) == batch_size:
                eeg_batch = torch.cat(batch_eeg_tensors, dim=0).unsqueeze(1).to(device)
                spec_batch = torch.cat(batch_spec_tensors, dim=0).to(device)

                optimizer.zero_grad()
                output = model(eeg_batch)
                loss = criterion(output, spec_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batch_eeg_tensors = []
                batch_spec_tensors = []

        if batch_eeg_tensors:
            eeg_batch = torch.cat(batch_eeg_tensors, dim=0).unsqueeze(1).to(device)
            spec_batch = torch.cat(batch_spec_tensors, dim=0).to(device)

            optimizer.zero_grad()
            output = model(eeg_batch)
            loss = criterion(output, spec_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / (len(eeg_X_shuffled) / batch_size)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.6f}')

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), output_path)
