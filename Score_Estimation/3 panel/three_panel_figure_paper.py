"""
Three-panel summary figure for the scalar reference system of the paper
"Intervention-Free Reconstruction of Forcing Histories in Nonlinear
Stochastic Systems" (Sec. 3.1 and Appendix A).

Panels:
  1. Representative trajectory of the reference dynamics, showing the
     noise-induced transitions between the two metastable regimes.
  2. Stationary scores s(x) = d/dx log rho_0(x): exact, Gaussian
     approximation, and the trained DSM + Stein network.
  3. Stationary densities rho_0(x) implied by each of the three scores,
     over the empirical histogram of the reference trajectory.

Reference dynamics (Eqs. 58-61) with the paper's parameter choice
F0 = 0, a = 1, b = 0, c = 1, sigma = 0.5:

    dX_t = (X_t - X_t^3) dt + sigma dW_t,     U(x) = x^4/4 - x^2/2,

a symmetric quartic double well with stable equilibria at x = +/-1 and an
unstable one at the origin, so rho_0 is bimodal and strongly non-Gaussian.
The invariant density and exact score are (Eq. 62)

    rho_0(x) = Z^-1 exp(-2 U(x) / sigma^2),   s_ex(x) = (2/sigma^2)(x - x^3),

and the Gaussian approximation (Eq. 63) is s_G(x) = -(x - mu_1)/v_0, which
matches the stationary mean and variance but cannot represent the nonlinear
score of a bimodal density. That failure is the point of panels 2 and 3: it
is the same misspecification that produces the spurious response kernels of
the paper's Fig. 3.

Numerics follow Appendix A: Euler-Maruyama with dt = 0.05 from X_0 = 0 for
N_ref = 500,000 steps (A.1); score network Linear(1->1024)/ReLU,
Linear(1024->1024)/tanh, Linear(1024->1), Adam at lr 1e-3, denoising scale
gamma = 0.05, mini-batches of 1024, 3000 epochs, polynomial Stein constraints
through degree K_S = 4 with unit weights and the quadratic penalty ramp
E0 = 0 -> E1 = 1000, lambda_max = 0.7 (A.2). After training, the same
identities are imposed exactly on the full reference sample by the degree-four
polynomial correction of Eqs. (19)-(21).

The DSM density in panel 3 is recovered by integrating the calibrated score
(cumulative trapezoid) to get log rho_0 up to a constant, then normalizing.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# numpy < 2 spells the quadrature np.trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz

SEED = 42

# Reference dynamics, Eq. (60): dX = (X - X^3) dt + sigma dW.
F0, A_LIN, B_QUAD, C_CUB = 0.0, 1.0, 0.0, 1.0
SIGMA = 0.5

# Appendix A.1
DT = 0.05
N_REF = 500_000
X_INIT = 0.0

# Appendix A.2
HIDDEN = 1024
EPOCHS = 3000
BATCH = 1024
LR = 1e-3
GAMMA = 0.05          # denoising scale
K_STEIN = 4           # polynomial Stein constraints through degree K_S
LAMBDA_MAX = 0.7
E0, E1 = 0, 1000      # quadratic ramp of the Stein penalty, Eq. (18)

N_TRAJ_PLOT = 4000    # steps shown in the trajectory panel

# Colors follow the paper's Fig. 3 convention.
C_EXACT, C_DSM, C_GAUSS = "tab:red", "tab:blue", "tab:green"


def generate_reference_trajectory():
    """Euler-Maruyama integration of Eq. (78)."""
    rng = np.random.default_rng(SEED)
    noise = SIGMA * np.sqrt(DT) * rng.standard_normal(N_REF)
    x = np.empty(N_REF)
    curr = X_INIT
    for n in range(N_REF):
        drift = F0 + A_LIN * curr + B_QUAD * curr**2 - C_CUB * curr**3
        curr = curr + drift * DT + noise[n]
        x[n] = curr
    return x


def potential(x):
    """U(x) = -F0 x - a x^2/2 - b x^3/3 + c x^4/4, Eq. (59)."""
    return -F0 * x - A_LIN * x**2 / 2 - B_QUAD * x**3 / 3 + C_CUB * x**4 / 4


def exact_score(x):
    """s_ex(x) = (2/sigma^2) (F0 + a x + b x^2 - c x^3), Eq. (62)."""
    return (2 / SIGMA**2) * (F0 + A_LIN * x + B_QUAD * x**2 - C_CUB * x**3)


def gaussian_score(x, mu1, v0):
    """s_G(x) = -(x - mu_1)/v_0, Eq. (63)."""
    return -(x - mu1) / v0


def normalize_density(x_grid, log_unnorm):
    log_d = log_unnorm - log_unnorm.max()
    density = np.exp(log_d)
    return density / _trapz(density, x_grid)


class ScoreNet(nn.Module):
    """Fig. 5: 1 -> 1024 (ReLU) -> 1024 (tanh) -> 1."""

    def __init__(self, size=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.Tanh(),
            nn.Linear(size, 1),
        )

    def forward(self, x):
        return self.net(x)


def stein_residuals(score, z, sigma_x, k_max=K_STEIN):
    """Empirical residuals of Eq. (16) for the test functions phi_k(x) = z^k.

    k = 0 enforces <s(X)>_0 = 0; k >= 1 enforces Eq. (81),
    <z^k s(X)>_0 = -(k/sigma_X) <z^(k-1)>_0.
    """
    residuals = [score.mean()]
    for k in range(1, k_max + 1):
        residuals.append((z**k * score).mean() + (k / sigma_x) * (z ** (k - 1)).mean())
    return residuals


def train(data, mu1, sigma_x):
    """DSM loss (Eq. 13) plus the ramped Stein penalty (Eqs. 17-18)."""
    torch.manual_seed(SEED)
    model = ScoreNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = data.size(0)

    for epoch in range(EPOCHS + 1):
        batch = data[torch.randint(0, n, (BATCH,))]

        # DSM term on noised samples: target is -Z/gamma with Z the raw noise.
        eps = torch.randn_like(batch) * GAMMA
        loss_dsm = torch.mean((model(batch + eps) + eps / GAMMA**2) ** 2)

        # Stein term on clean samples, unit weights w_k = 1.
        z = (batch - mu1) / sigma_x
        residuals = stein_residuals(model(batch), z, sigma_x)
        loss_stein = sum(r**2 for r in residuals)

        ramp = min(max((epoch - E0) / (E1 - E0), 0.0), 1.0)
        lambda_stein = LAMBDA_MAX * ramp**2

        loss = loss_dsm + lambda_stein * loss_stein
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 250 == 0:
            print(
                f"epoch {epoch:5d} | DSM {loss_dsm.item():10.4f} | "
                f"Stein {loss_stein.item():.6f} | lambda {lambda_stein:.4f}",
                flush=True,
            )
    return model


def stein_calibration(model, data, mu1, sigma_x):
    """Degree-four polynomial correction of Eqs. (19)-(21).

    Returns a callable score s_cal(x) = s_theta(x) + sum_j kappa_j z^j whose
    empirical Stein residuals on the full reference sample vanish to numerical
    tolerance. The Gram matrix of Eq. (21) collects standardized moments up to
    order 2*K_S and is ill conditioned, so it is solved in the least-squares
    (pseudoinverse) sense as the paper prescribes.
    """
    with torch.no_grad():
        s_theta = model(data).numpy().flatten()
    x = data.numpy().flatten()
    z = (x - mu1) / sigma_x

    phi = np.stack([z**k for k in range(K_STEIN + 1)])            # (K+1, N)
    # d/dx phi_k = k z^(k-1) / sigma_x
    dphi = np.stack(
        [np.zeros_like(z)] + [k * z ** (k - 1) / sigma_x for k in range(1, K_STEIN + 1)]
    )

    gram = phi @ phi.T / len(z)
    rhs = -(dphi + phi * s_theta).mean(axis=1)
    kappa, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

    def s_cal(x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        z_new = (x_new - mu1) / sigma_x
        with torch.no_grad():
            base = model(
                torch.tensor(x_new, dtype=torch.float32).view(-1, 1)
            ).numpy().flatten()
        return base + sum(kappa[k] * z_new**k for k in range(K_STEIN + 1))


    return s_cal, kappa


def report_residuals(label, score_vals, z, sigma_x):
    res = stein_residuals(score_vals, z, sigma_x)
    print(f"{label} Stein residuals: {[float(f'{r:.3e}') for r in res]}", flush=True)
    return res


def main():
    print("integrating reference trajectory...", flush=True)
    x_np = generate_reference_trajectory()
    data = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)

    mu1 = float(x_np.mean())
    v0 = float(x_np.var())
    sigma_x = float(np.sqrt(v0))
    print(f"mu_1 = {mu1:.5f} | v_0 = {v0:.5f} | sigma_X = {sigma_x:.5f}", flush=True)

    model = train(data, mu1, sigma_x)
    s_cal, kappa = stein_calibration(model, data, mu1, sigma_x)
    print(f"calibration coefficients kappa = {np.array2string(kappa, precision=4)}", flush=True)

    # Sanity check that the calibration did what Eq. (20) claims, on the full sample.
    z_full = (x_np - mu1) / sigma_x
    with torch.no_grad():
        s_raw = model(data).numpy().flatten()
    report_residuals("pre-calibration ", s_raw, z_full, sigma_x)
    report_residuals("post-calibration", s_cal(x_np), z_full, sigma_x)

    # Grid over the sampled region, trimmed to where the tails are populated.
    x_max = float(np.quantile(np.abs(x_np), 0.9995))
    x_grid = np.linspace(-x_max, x_max, 601)

    s_ex = exact_score(x_grid)
    s_g = gaussian_score(x_grid, mu1, v0)
    s_dsm = s_cal(x_grid)

    pdf_exact = normalize_density(x_grid, -2 * potential(x_grid) / SIGMA**2)
    pdf_gauss = np.exp(-((x_grid - mu1) ** 2) / (2 * v0)) / np.sqrt(2 * np.pi * v0)
    pdf_dsm = normalize_density(x_grid, cumulative_trapezoid(s_dsm, x_grid, initial=0))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    t = np.arange(N_TRAJ_PLOT) * DT
    axes[0].plot(t, x_np[:N_TRAJ_PLOT], lw=0.7, color="black")
    for well in (-1.0, 1.0):
        axes[0].axhline(well, color="gray", lw=0.8, ls="--")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$X_t$")
    axes[0].set_title(rf"Reference trajectory ($\sigma={SIGMA}$, $\Delta t={DT}$)")

    axes[1].plot(x_grid, s_ex, color=C_EXACT, lw=2.5, label=r"exact $s_{\rm ex}$")
    axes[1].plot(x_grid, s_dsm, color=C_DSM, ls="--", lw=1.8, label=r"DSM $s_\theta$ (calibrated)")
    axes[1].plot(x_grid, s_g, color=C_GAUSS, ls=":", lw=1.8, label=r"Gaussian $s_G$")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$s(x)$")
    axes[1].set_title("Stationary score")
    axes[1].legend()

    axes[2].hist(
        x_np, bins=120, range=(-x_max, x_max), density=True,
        color="lightgray", label="empirical", zorder=0,
    )
    axes[2].plot(x_grid, pdf_exact, color=C_EXACT, lw=2.5, label=r"exact $\rho_0$")
    axes[2].plot(x_grid, pdf_dsm, color=C_DSM, ls="--", lw=1.8, label="DSM")
    axes[2].plot(x_grid, pdf_gauss, color=C_GAUSS, ls=":", lw=1.8, label="Gaussian")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$\rho_0(x)$")
    axes[2].set_title("Stationary density (bimodal)")
    axes[2].legend()

    fig.tight_layout()
    out = "three_panel_figure_paper_with_2_layers.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}", flush=True)

    # Score error over the sampled region, for the record.
    for name, s in (("DSM", s_dsm), ("Gaussian", s_g)):
        w = pdf_exact / _trapz(pdf_exact, x_grid)
        rmse = np.sqrt(_trapz(w * (s - s_ex) ** 2, x_grid))
        print(f"{name} density-weighted score RMSE = {rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
