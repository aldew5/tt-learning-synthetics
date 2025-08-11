import torch
import torch.nn as nn
import torch.optim as optim
from utils import train, MLP, parser
import matplotlib.pyplot as plt
from constants import LR
import ivon

if __name__ == "__main__":
    args = parser.parse_args()

    d, t = 32, 10000
    torch.manual_seed(0)

    # TODO: by default, weights are uniform random. Atlas doesn't say how they sample 
    teacher = MLP(d, d).eval()      
    for p in teacher.parameters():
        p.requires_grad_(False)

    I = torch.randn(t, d)
    O = teacher(I) 
    # normalize the dataset statistics
    O = (O) / O.std()

    M_learn = MLP(d, d)
    if not args.ivon:
        optimizer = optim.AdamW(M_learn.parameters(), lr=LR)
    else:
        optimizer = ivon.IVON(M_learn.parameters(), lr=0.1, ess=len(I))

    losses = train(M_learn, optimizer, I, O, args.ivon)
    print("Final loss:", losses[-1])

    plt.plot(losses)
    plt.xlabel("Step j")
    plt.ylabel("Normed MSE loss_j")
    plt.title("Online Learn of MLP Mapping")
    plt.show()
