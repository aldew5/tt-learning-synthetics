import torch
import torch.nn as nn
import torch.optim as optim
from utils import train, MLP, parser
import matplotlib.pyplot as plt
from constants import LR
import ivon

"""
This is a new synthetic setting 
for testing the ability of IVON to deal with surprising data.
"""


if __name__ == "__main__":
    args = parser.parse_args()

    d, t = 32, 10000
    FROM_TEACHER = int(t * 0.5)
    torch.manual_seed(0)

    # TODO: by default, weights are uniform random. Atlas doesn't say how they sample 
    teacher = MLP(d, d).eval()     
    teacher_surprise = MLP(d, d).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in teacher_surprise.parameters():
        p.requires_grad_(False)
    
    I_teacher = torch.randn(t, d)
    I_surprise = torch.randn(t, d)
    O_teacher = teacher(I_teacher)
    O_surprise = teacher_surprise(I_surprise)

    O_teacher = (O_teacher) / O_teacher.std()
    O_surprise = (O_surprise) / O_surprise.std()

    O_final = []
    I_final = []
    
    O_final.append(O_teacher[:FROM_TEACHER])
    I_final.append(I_teacher[:FROM_TEACHER])
    
    # Last 20%: 50% chance from teacher, 50% chance from surprise
    MIXED = t - FROM_TEACHER
    print("MIXED", MIXED)
    for i in range(MIXED):
        if torch.rand(1) < 0.5:
            # From teacher (use remaining teacher samples)
            idx = FROM_TEACHER + i
            O_final.append(O_teacher[idx:idx+1])
            I_final.append(I_teacher[idx:idx+1])
        else:
            # From surprise (use remaining surprise samples)
            idx = i
            O_final.append(O_surprise[idx:idx+1])
            I_final.append(I_surprise[idx:idx+1])
    
    print(len(O_final))
    # Concatenate all parts 
    O = torch.cat(O_final, dim=0)
    I = torch.cat(I_final, dim=0)
    
    # Create a random permutation of the row indices
    perm = torch.randperm(MIXED)
    
    # Apply the permutation only to the last MIXED indices of both O and I
    O[FROM_TEACHER:] = O[FROM_TEACHER:][perm]
    I[FROM_TEACHER:] = I[FROM_TEACHER:][perm]



    M_learn = MLP(d, d)
    ESS = len(I)/500
    name = "AdamW" if not args.ivon else "IVON"
    if not args.ivon:
        optimizer = optim.AdamW(M_learn.parameters(), lr=LR)
    else:
        optimizer = ivon.IVON(M_learn.parameters(), lr=0.1, ess=ESS)

    losses = train(M_learn, optimizer, I, O, args.ivon)
    print("Final loss:", losses[-1])

    plt.plot(losses)
    plt.xlabel("Step j")
    plt.ylabel("Normed MSE loss_j")
    plt.title(f"optimizer={name}, d={d}, t={t}, ess={ESS}")
    plt.show()
