import torch.nn as nn
import argparse
import torch.nn.functional as F

def normalized_mse(pred, o):
    """
    Normalized MSE: ||pred - o||^2 / ||o||^2
    """
    return ((pred - o).pow(2).sum()
            / (o.pow(2).sum().clamp_min(1e-6)))

def train(model, optimizer, inputs, targets, ivon=False):
    """
    inputs:  [t × d] tensor of i_j
    targets: [t × d] tensor of o_j
    """
    model.train()
    losses = []
    train_samples = 1
    if not ivon:
        print("Training with AdamW")
        for i, o in zip(inputs, targets):
            pred = model(i)
            loss = normalized_mse(pred, o)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    else:
        print("Training with IVON")
        for i, o in zip(inputs, targets):
            for _ in range(train_samples):
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    pred = model(i)
                    loss = normalized_mse(pred, o)
                    loss.backward()

            optimizer.step()
            losses.append(loss.item())

    return losses

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)


parser = argparse.ArgumentParser(description='Online learning experiment')
parser.add_argument(
    '--ivon',
    action='store_true',
    help='run IVON instead of AdamW'
)