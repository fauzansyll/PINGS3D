import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import Tuple

# ===============================
# Reproducibility
# ===============================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ===============================
# 3D ANISOTROPIC DATASET
# ===============================
def sample_3d_anisotropic(n: int, shift=(0, 0, 0)) -> np.ndarray:
    modes = [
        np.array([2.0, 0.0, 0.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 1.0])
    ]
    covs = [
        np.diag([0.4, 0.05, 0.05]),
        np.diag([0.2, 0.3, 0.05]),
        np.diag([0.1, 0.1, 0.5])
    ]

    xs = []
    for _ in range(n):
        k = np.random.randint(0, len(modes))
        x = np.random.multivariate_normal(modes[k], covs[k])
        xs.append(x)

    X = np.stack(xs)
    return X + np.array(shift)

# ===============================
# PINGS-STYLE MODEL
# ===============================
class PINGS(nn.Module):
    def __init__(self, dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)

# ===============================
# KERNEL MMD (NUMPY ONLY)
# ===============================
def kernel_mmd(x: np.ndarray, y: np.ndarray, sigma=1.0) -> float:
    def k(a, b):
        sq = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
        return np.exp(-sq / (2 * sigma ** 2))

    kxx = k(x, x).mean()
    kyy = k(y, y).mean()
    kxy = k(x, y).mean()
    return kxx + kyy - 2 * kxy

# ===============================
# PCA (NUMPY SVD)
# ===============================
def pca_numpy(X: np.ndarray, n_components=2) -> np.ndarray:
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:n_components].T

# ===============================
# TRAINING LOOP
# ===============================
def train(model, data, epochs=2000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_target = torch.tensor(data, dtype=torch.float32)

    for _ in range(epochs):
        x0 = torch.randn_like(x_target)
        v = model(x0)
        x1 = x0 + v
        loss_flow = ((x1 - x_target) ** 2).mean()
        loss_endpoint = ((model(x_target)) ** 2).mean()
        loss = loss_flow + 0.05 * loss_endpoint

        opt.zero_grad()
        loss.backward()
        opt.step()

# ===============================
# SAMPLING
# ===============================
@torch.no_grad()
def sample(model, n=2000, steps=100):
    x = torch.randn(n, 3)
    for _ in range(steps):
        x = x + model(x) * 0.05
    return x.numpy()

# ===============================
# TRAJECTORY GIF
# ===============================
@torch.no_grad()
def animate_trajectories(model, save_path="trajectories.gif"):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    x = torch.randn(200, 3)

    frames = []
    for t in range(50):
        ax.clear()
        x = x + model(x) * 0.05
        xp = x.numpy()

        ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], s=6)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        ax.set_title(f"Step {t}")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())

    plt.close(fig)
    imageio.mimsave(save_path, frames, fps=10)

# ===============================
# MAIN EXPERIMENT
# ===============================
def main():
    seeds = [0, 1, 2, 3, 4]
    mmds = []

    train_data = sample_3d_anisotropic(2000)
    test_data = sample_3d_anisotropic(2000, shift=(0.5, -0.3, 0.2))

    for seed in seeds:
        set_seed(seed)
        model = PINGS()
        train(model, train_data)
        samples = sample(model)

        mmd = kernel_mmd(samples, test_data)
        mmds.append(mmd)

        if seed == 0:
            animate_trajectories(model)

            # 3D scatter
            fig = plt.figure(figsize=(12, 4))
            for i, (title, X) in enumerate([
                ("Train", train_data),
                ("Test", test_data),
                ("Generated", samples)
            ]):
                ax = fig.add_subplot(1, 3, i + 1, projection="3d")
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=4)
                ax.set_title(title)
            plt.show()

            # PCA
            proj = pca_numpy(samples)
            plt.scatter(proj[:, 0], proj[:, 1], s=4)
            plt.title("PCA Projection (Appendix)")
            plt.show()

    print("MMD mean ± std:", np.mean(mmds), np.std(mmds))

    plt.errorbar(range(len(seeds)), mmds, yerr=np.std(mmds))
    plt.xlabel("Seed")
    plt.ylabel("MMD")
    plt.title("Stability Across Seeds")
    plt.show()

if __name__ == "__main__":
    main()
