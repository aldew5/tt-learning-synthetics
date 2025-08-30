import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

def normalized_mse(pred, o):
    """
    Normalized MSE: ||pred - o||^2 / ||o||^2
    """
    return ((pred - o).pow(2).sum()
            / (o.pow(2).sum().clamp_min(1e-6)))

@torch.no_grad()
def _predictive_stats_mc(model, optimizer, x, T: int = 30):
    """
    Monte-Carlo predictive mean/variance under IVON's parameter posterior.
    Uses optimizer.sampled_params(train=False) as a context manager.
    Returns: mean (d,), var (d,)
    """
    preds = []
    for _ in range(T):
        with optimizer.sampled_params(train=False):
            preds.append(model(x).detach())
    preds = torch.stack(preds, dim=0)   
    mu = preds.mean(dim=0) 
    var = preds.var(dim=0, unbiased=False) 
    return mu, var

@torch.no_grad()
def nlpl_ivon_mc(model, optimizer, x, y, sigma_obs: float = 1e-2, T: int = 30):
    """
    Compute NLPL(v|k,q) = 0.5 * sum_i [ (y_i - mu_i)^2 / (var_i + sigma_obs^2) + log(var_i + sigma_obs^2) ]
    using MC predictive statistics from IVON.

    Returns:
      nlpl (scalar float),
      pred_std (scalar float): mean predictive std (useful for your 'stds' list)
    """
    mu, var = _predictive_stats_mc(model, optimizer, x, T=T)
    var_tot = var + (sigma_obs ** 2)
    # numerical floor for safety
    var_tot = torch.clamp(var_tot, min=1e-20)

    err = y - mu
    maha = (err * err) / var_tot
    nlpl = 0.5 * (maha.sum() + torch.log(var_tot).sum())
    pred_std = torch.sqrt(var).mean()  # or torch.sqrt(var_tot).mean() if you want obs-noise included
    return nlpl.item(), pred_std.item()


def train(model, optimizer, inputs, targets, stds=None, nlpls=None, ivon=False):
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
        cnt = 0
        for i, o in zip(inputs, targets):
            for _ in range(train_samples):
                with optimizer.sampled_params(train=True):
                    optimizer.zero_grad()
                    pred = model(i)
                    loss = normalized_mse(pred, o)
                    loss.backward()
            cnt += 1

            optimizer.step()
            losses.append(loss.item())
            nlpl, pred_std = nlpl_ivon_mc(model, optimizer, i, o, sigma_obs=1e-2, T=30)
            nlpls.append(nlpl)
            stds.append(pred_std)
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

def mse_vk(M, k, v):
    """Normalized MSE"""
    pred = torch.einsum("vk,btk->btv", M, k)
    return ((pred - v).pow(2).sum()
            / (v.pow(2).sum().clamp_min(1e-6)))

def cos_M(M, M_star, eps=1e-10):
    num = (M * M_star).sum()
    den = (M.norm() * M_star.norm() + eps)
    return float(num / den)
