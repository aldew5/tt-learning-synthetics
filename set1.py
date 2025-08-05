import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import train, MLP, parser
from constants import LR
import ivon

def make_low_rank_data(d, k, t, seed=None, device='cpu'):
    """
    Generates
      X ∈ R^{d×k}, Y ∈ R^{k×d}, W = X @ Y (so W is d×d low–rank)
      i_1,…,i_t ∼ N(0,I_d)
      o_j = W^T @ i_j  for j=1…t
    Returns:
      I: array of shape (t, d)  — the i_j’s
      O: array of shape (t, d)  — the o_j’s
    """
    if seed is not None:
        torch.manual_seed(seed)
    X = torch.randn(d, k, device=device)
    Y = torch.randn(k, d, device=device)
    W = X @ Y
    I = torch.randn(t, d, device=device)
    O = I @ W
    return I, O

if __name__ == "__main__":
    args = parser.parse_args()

    # NOTE: k is not specified in ATLAS. We choose k st loss is similar to baseline
    d, k, t = 256, 128, 10000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data
    I, O = make_low_rank_data(d, k, t, seed=42, device=device)
    print("Standard deviation of O:", O.std())
    # Normalize the dataset statistics
    O = (O) / O.std()

    # NOTE: we keep the hidden layer dimension the same. This corresponds to fig 6a
    model = MLP(d).to(device)

    if not args.ivon:
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
    else:
        optimizer = ivon.IVON(model.parameters(), lr=0.14, ess=len(I) * 0.5)

    # run online training
    losses = train(model, optimizer, I, O, args.ivon)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step (j)")
    plt.ylabel("Normalized MSE Loss_j")
    plt.title("Online Training Loss over Time")
    plt.show()

    # inspect
    print(f"Final loss: {losses[-1]:.3e}")