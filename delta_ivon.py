import math, torch
import matplotlib.pyplot as plt
from utils import mse_vk, cos_M

def make_synth_memory(B=8, T=64, K=32, V=32, noise=0, device="cuda"):
    """Randomly generate a synthetic memory baseline"""
    M_star = torch.randn(V, K, device=device) / math.sqrt(K)
    # TODO: random keys okay for this test but need structure for surprise
    k = torch.randn(B, T, K, device=device)
    v = torch.einsum("vk,btk->btv", M_star, k)

    # optionally use a noisy values dataset
    if noise > 0:
        v = v + noise * torch.randn_like(v)
    return k, v, M_star

@torch.no_grad()
def ivon_delta_step(k_t, v_t, M, g, h, *,
                    lr=1e-3, beta1=0.9, rho=0.999,
                    delta=1e-4, lam=100):
    """
    Single head (V x K). Dela rule IVON. Recompute Sigma each time instead of 
    passing it as an argument.

    Note that lam = N = "effective sample size"
    """
    # additional stability param
    eps = 1e-10

    # delta rule
    v_hat = torch.einsum("vk,bk->bv", M, k_t)
    err = v_t - v_hat
    # average stochastic grad
    g_hat = -torch.einsum("bv,bk->vk", err, k_t) / k_t.shape[0]

    # curvature
    denom = lam * (h + delta) + eps
    sigma = denom.rsqrt()
    z = torch.randn_like(M)
    h_hat = g_hat * (z / (sigma + eps)) 

    g.mul_(beta1).add_(g_hat, alpha=(1 - beta1))
    diff = (h - h_hat)
    corr = 0.5 * (1 - rho)**2 * (diff * diff) / (h + delta + eps)
    h.mul_(rho).add_(h_hat, alpha=(1 - rho)).add_(corr)

    # update mean
    M.addcdiv_(g + delta * M, (h + delta + eps), value=-lr)

def run_test(epochs=100, log=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B,T,K,V = 8,64,32,32
    k, v, M_star = make_synth_memory(B,T,K,V, noise=0, device=device)

    M = torch.randn(V,K,device=device)/math.sqrt(K)
    g, h = torch.zeros_like(M), torch.zeros_like(M)

    print("=== IVON Dela Rule ===")
    losses, cosines = [], []
    for epoch in range(epochs):
        for t in range(T):
            ivon_delta_step(k[:,t], v[:,t], M, g, h,
                            lr=0.01, beta1=0.9, rho=0.999,
                            delta=1e-4, lam=T)
        L, C = mse_vk(M,k,v), cos_M(M,M_star)
        losses.append(L); cosines.append(C)
        
        if log:
            print(f"epoch {epoch:02d}  MSE={L:.6f}  cos={C:.4f}")
    
    # sanity check
    assert losses[-1] < losses[0]
    return losses

if __name__ == "__main__":
    losses = run_test(epochs=1000000)
    plt.plot(losses)
    plt.xlabel("Step j")
    plt.ylabel("Normed MSE loss_j")
    plt.title("Online Learn of MLP Mapping")
    plt.show()

