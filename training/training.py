import torch
from torch.utils.data import DataLoader


def train_epoch(model, dataset, optimizer, criterion):

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()

    total_loss = 0

    for x, y in loader:

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
