"""DSM on stochastic Lorenz 63, evaluated by score-driven Langevin sampling.

Install: pip install numpy scipy matplotlib torch
Run: python lorenz63_dsm.py
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch import nn
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.ndimage import gaussian_filter
from plotly.subplots import make_subplots
import plotly.graph_objects as go

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All simulations use dimensionless Lorenz units.
SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DYNAMICAL_NOISE = 0.5  # additive SDE diffusion coefficient
DT = 0.005
N_STEPS = 260000
BURN_IN = 20000
THIN = 10
DSM_NOISE = 0.15  # standard deviation in standardized coordinates
EPOCHS = 120
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
LANGEVIN_STEP = 0.001
LANGEVIN_STEPS = 40000
LANGEVIN_BURN_IN = 10000
LANGEVIN_THIN = 10
N_CHAINS = 64


def lorenz_drift(x):
    a, b, c = x
    return np.array([SIGMA * (b - a), a * (RHO - c) - b, a * b - BETA * c])


def simulate_lorenz():
    """RK4 deterministic drift with additive Euler noise per step."""
    rng = np.random.default_rng(SEED)
    x = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    samples = []
    for t in range(N_STEPS):
        print(f'\rSimulating Lorenz 63: step {t+1}/{N_STEPS}', end='', flush=True)
        k1 = lorenz_drift(x)
        k2 = lorenz_drift(x + DT * k1 / 2)
        k3 = lorenz_drift(x + DT * k2 / 2)
        k4 = lorenz_drift(x + DT * k3)
        x = x + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += DYNAMICAL_NOISE * np.sqrt(DT) * rng.standard_normal(3)
        if t >= BURN_IN and (t - BURN_IN) % THIN == 0:
            samples.append(x.copy())
    return np.asarray(samples, dtype=np.float32)


class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


def train_dsm(data):
    model = ScoreNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_data = torch.as_tensor(data, device=DEVICE)
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(train_data), device=DEVICE)
        total = 0.0
        for idx in perm.split(BATCH_SIZE):
            clean = train_data[idx]
            eps = torch.randn_like(clean)
            noisy = clean + DSM_NOISE * eps
            # Score of Gaussian corruption: -(noisy-clean)/sigma^2
            target = -eps / DSM_NOISE
            pred = model(noisy)
            loss = ((pred - target) ** 2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{EPOCHS}: DSM loss {total/len(train_data):.4f}', flush=True)
    return model


@torch.no_grad()
def sample_langevin(model, initial):
    """Overdamped Langevin with drift score and diffusion sqrt(2).

    Invariant distribution in the infinitesimal-step limit is the density
    whose score is learned by model. Finite steps introduce bias.
    """
    model.eval()
    x = torch.as_tensor(initial, dtype=torch.float32, device=DEVICE).clone()
    draws = []
    for step in range(LANGEVIN_STEPS):
        score = model(x)
        x += LANGEVIN_STEP * score + np.sqrt(2 * LANGEVIN_STEP) * torch.randn_like(x)
        if step >= LANGEVIN_BURN_IN and (step - LANGEVIN_BURN_IN) % LANGEVIN_THIN == 0:
            draws.append(x.cpu().numpy().copy())
    return np.concatenate(draws, axis=0)


def evaluate(reference, generated):
    print('\nStationary distribution comparison (physical Lorenz coordinates)')
    print('Variable | reference mean | generated mean | reference std | generated std | Wasserstein-1')
    for j, name in enumerate(['x', 'y', 'z']):
        print(f'{name:8s} | {reference[:,j].mean():14.3f} | {generated[:,j].mean():14.3f} | '
              f'{reference[:,j].std():13.3f} | {generated[:,j].std():13.3f} | '
              f'{wasserstein_distance(reference[:,j], generated[:,j]):13.3f}')
    print('Reference covariance:\n', np.cov(reference.T))
    print('Generated covariance:\n', np.cov(generated.T))


def plot_results(reference, generated):
    fig = plt.figure(figsize=(12, 5))
    for i, (data, title) in enumerate([(reference, 'Original stochastic Lorenz 63'),
                                       (generated, 'DSM score Langevin samples')], 1):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        sub = data[::max(1, len(data)//6000)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.5, alpha=0.25)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=title)
    fig.tight_layout()
    fig.savefig('lorenz63_3d.png', dpi=180)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for j, name in enumerate(['x', 'y', 'z']):
        axes[j].hist(reference[:, j], bins=75, density=True, alpha=0.55, label='Original')
        axes[j].hist(generated[:, j], bins=75, density=True, alpha=0.55, label='DSM Langevin')
        axes[j].set(xlabel=name, ylabel='Density', title=f'Marginal distribution: {name}')
    axes[0].legend()
    fig.tight_layout()
    # fig.savefig('lorenz63_marginals.png', dpi=180)
    plt.show()


def plot_joint_density_3d(
    original,
    reconstructed,
    grid_size=40,
    max_samples=10000,
    bandwidth=None,
    output_file="lorenz63_joint_density.html"
):
    """
    Compare original and DSM reconstructed 3D joint densities.

    Parameters
    ----------
    original : ndarray (N, 3)
        Original Lorenz 63 states.
    reconstructed : ndarray (M, 3)
        States generated using DSM Langevin dynamics.
    grid_size : int
        Number of grid points per dimension.
    max_samples : int
        Maximum samples used to fit each KDE.
    bandwidth : float or None
        KDE bandwidth factor. None uses Scott's rule.
    output_file : str
        HTML output filename.
    """

    rng = np.random.default_rng(42)

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    def subsample(data):
        if len(data) > max_samples:
            indices = rng.choice(len(data), max_samples, replace=False)
            return data[indices]
        return data

    original = subsample(original)
    reconstructed = subsample(reconstructed)

    combined = np.vstack([original, reconstructed])

    lower = np.percentile(combined, 0.5, axis=0)
    upper = np.percentile(combined, 99.5, axis=0)

    padding = 0.1 * (upper - lower)
    lower -= padding
    upper += padding

    x = np.linspace(lower[0], upper[0], grid_size)
    y = np.linspace(lower[1], upper[1], grid_size)
    z = np.linspace(lower[2], upper[2], grid_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    positions = np.vstack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])

    def estimate_density(data):
        kde = gaussian_kde(data.T, bw_method=bandwidth)
        return kde(positions).reshape(X.shape)

    density_original = estimate_density(original)
    density_dsm = estimate_density(reconstructed)

    # Use the same density thresholds for both plots.
    maximum = max(density_original.max(), density_dsm.max())
    minimum = 0.10 * maximum

    

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Original Lorenz 63",
            "DSM Reconstruction"
        ]
    )

    for col, density in enumerate(
        [density_original, density_dsm], start=1
    ):
        fig.add_trace(
            go.Isosurface(
                x=X.ravel(),
                y=Y.ravel(),
                z=Z.ravel(),
                value=density.ravel(),
                isomin=minimum,
                isomax=maximum,
                surface_count=5,
                opacity=0.25,
                colorscale="Viridis",
                showscale=(col == 2),
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                )
            ),
            row=1,
            col=col
        )

    fig.update_layout(
        title="Lorenz 63 Joint Probability Density",
        width=1200,
        height=600,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        ),
        scene2=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        )
    )

    fig.write_html(output_file)
    fig.show()

    return density_original, density_dsm
def main():
    print('Device:', DEVICE, flush=True)
    reference = simulate_lorenz()
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    standardized = (reference - mean) / std
    model = train_dsm(standardized)
    # In standardized coordinates, DSM learns the Gaussian-smoothed distribution.
    # Add the same smoothing to the reference when testing the learned target.
    rng = np.random.default_rng(SEED + 1)
    smoothed_ref = standardized + DSM_NOISE * rng.standard_normal(standardized.shape)
    initial = standardized[rng.choice(len(standardized), N_CHAINS, replace=True)]
    generated = sample_langevin(model, initial)
    reference_physical = smoothed_ref * std + mean
    generated_physical = generated * std + mean
    print('\nEvaluating results...', flush=True)
    evaluate(reference_physical, generated_physical)
    print('\nSaving model and plots...', flush=True)
    plot_results(reference_physical, generated_physical)
    torch.save({'model_state_dict': model.cpu().state_dict(),
                'mean': mean, 'std': std, 'dsm_noise': DSM_NOISE}, 'lorenz63_dsm.pt')
    plot_joint_density_3d(reference_physical, generated_physical)
    print('\nSaved model and plots.')


if __name__ == '__main__':
    main()
