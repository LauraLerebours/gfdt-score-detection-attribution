"""
Three-panel summary figure for the 1D score-estimation model:
  1. A representative trajectory of the simulated SDE.
  2. Score functions s(x) = d/dx log(rho(x)) from three methods:
     analytical (exact, from the known drift), Gaussian (fit to the
     trajectory's mean/variance), and DSM (the trained score network).
  3. Stationary PDFs implied by each of the three methods, with the
     empirical histogram of the trajectory shown for reference.

Analytical score/PDF: for dx = F(x) dt + sigma dW with F(x) = F + a*x +
b*x^2 - c*x^3, the stationary density satisfies rho(x) ~ exp((2/sigma^2)
* V(x)) where V(x) = int F(x) dx. So s(x) = (2/sigma^2) * F(x) exactly.

The system is the reference model of the paper (Sec. 3.1, Eqs. 58-62):
F = 0, a = 1, b = 0, c = 1, sigma = 0.5, i.e.

    dx = (x - x^3) dt + sigma dW,     U(x) = x^4/4 - x^2/2,

a symmetric quartic double well with stable equilibria at x = +/-1 and an
unstable one at the origin. Noise drives transitions between the two wells,
so rho is bimodal and strongly non-Gaussian and the exact score
s(x) = (2/sigma^2)(x - x^3) is cubic. The Gaussian fit matches only the
stationary mean and variance, so it cannot represent this score at all -
its failure in panels 2 and 3 is the point of the figure, and it is the
same misspecification that produces the spurious response kernels in the
paper's Fig. 3. (Setting b = c = 0 with a < 0 instead gives an
Ornstein-Uhlenbeck process, whose rho is exactly Gaussian; that variant is
a sanity check, not the paper's system.)

Sampling follows the paper's Appendix A.1 rather than the defaults in
dsm_with_stein: dt = 0.05, 500,000 steps from X0 = 0 with no burn-in, i.e.
25,000 time units per trajectory. This matters more than it looks. Escaping
a well is a Kramers barrier crossing, not a fast relaxation, so the
integrated autocorrelation time is tau_int ~ 29 time units (574 steps) and a
single A.1 trajectory carries only ~870 effective samples despite having
500,000 of them. (The module defaults, dt = 0.01 and 100,000 steps = 1,000
time units, give ~35 and produce a badly lopsided histogram.)

Even at A.1 length one trajectory makes only ~740 well-to-well switches, so
the occupancy fraction P(x>0) has a standard error of sqrt(0.25/740) ~ 0.018
and the two modes of the empirical histogram routinely differ in height by
~7%, occasionally by 17%, purely from sampling noise. The symmetry of rho is
a property of the exact stationary density; no finite trajectory inherits
it. So instead of one trajectory we integrate an ENSEMBLE of N_REPLICATES
independent ones, each following A.1 exactly, vectorized across replicates
in numpy (the whole ensemble costs seconds, versus minutes for one scalar
torch loop). The histogram, the moments and the Gaussian fit use all of
them, which cuts the symmetry error by sqrt(N_REPLICATES).

N_eff, not the sample count, is what the network can actually learn from,
and N_eff = (total simulated time)/tau_int is linear in N_REPLICATES and
completely independent of how densely each trajectory is stored. So the
ensemble is stored THINNED, one state every THIN ~ 0.44 tau_int steps.
Thinning discards redundancy rather than information - it improves no
estimate on its own - but it cuts memory 250x, which is what makes 256
replicates affordable. The result is a training set the same size as the
paper's single-trajectory budget, so d.train costs exactly what it did,
carrying ~10x the effective sample size.

This matters because DSM is fitting the EMPIRICAL density. Autocorrelation
does not bias the DSM gradient - correlated draws are still marginally
distributed as rho - but it inflates the error of the empirical density,
and the network then reproduces that error faithfully. No amount of
reshuffling or resampling can undo it; only more independent chains can.

The DSM curve is the score the paper's D&A sweep actually uses, which is NOT
the one used for the Sec. 3.3 kernels. Appendix A.2 defines two configs; see
CONFIGS below. CONFIG = 'attribution' reproduces Fig. 1's score: 2000 epochs
of DSM loss alone, then the affine calibration of Eq. (80). CONFIG =
'kernels' reproduces Sec. 3.3's: 3000 epochs with the ramped Stein penalty
through K_S = 4, then a degree-four correction.

Both end in the post-training projection of Eqs. (19)-(21), which is what
makes the plotted score usable in a response kernel: DSM learns the score of
a gamma-smoothed density (Eq. 14), and the residual global bias that leaves
behind is precisely what contaminates the correlations in Eq. (9). Since the
density is exp(int s), a score bias the eye cannot see in panel 2 becomes a
visible distortion in panel 3.

DSM PDF: recovered by numerically integrating the learned score
(cumulative trapezoid) to get log rho up to a constant, then normalizing.
"""
import hashlib
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
import torch
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

import dsm_with_stein as d

# numpy < 2 spells the quadrature np.trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz

# One root seed for the whole experiment. Set to None for a fresh draw: the
# entropy numpy actually chose is recorded in the run's metadata sidecar, so
# even an unseeded run can be replayed exactly by pasting it back here.
SEED = 123
ROOT_SS = np.random.SeedSequence(SEED)

# Independent substreams, one per consumer. Deriving them from a single root
# rather than reusing SEED means the ensemble, the training-subset draw and
# torch cannot consume each other's random numbers, so changing one of them
# leaves the others bit-for-bit unchanged.
SS_ENSEMBLE, SS_SUBSET, SS_TORCH = ROOT_SS.spawn(3)

# Paper Sec. 3.1, Eq. (60): quartic double well.
F, a, b, c = 0.0, 1.0, 0.0, 1.0
assert c > 0.0 and a > 0.0, "bimodal rho requires a > 0 (unstable origin) and c > 0"

# Paper Appendix A.1. Deliberately overrides d.dt / d.num_steps; see docstring.
DT = 0.05
N_REF = 500_000

# Independent trajectories, each of length N_REF. See docstring: this is what
# makes the empirical PDF symmetric, and it is the only knob that adds
# information about rho. N_eff scales linearly with it.
N_REPLICATES = 256

# Integrated autocorrelation time, measured for this system at DT = 0.05.
TAU_STEPS = 574

# Keep every THIN-th step. At ~0.44 tau_int consecutive kept samples are
# nearly independent, so this discards redundancy rather than information:
# it is what makes 256 replicates affordable (512k samples retained out of
# 128M integrated). Thinning does NOT by itself improve any estimate.
THIN = 250

# Wiener increments are drawn in blocks of this many steps so peak memory is
# NOISE_BLOCK x N_REPLICATES rather than N_REF x N_REPLICATES. Each replicate
# keeps its own Generator across blocks, so the stream - and therefore the
# trajectory - is identical to drawing all N_REF at once.
NOISE_BLOCK = 50_000

# 4000 steps * DT = 200 time units, ~7 correlation times, so the trajectory
# panel actually shows the well-to-well transitions that make rho bimodal.
N_TRAJ_PLOT = 4000

# Colors follow the paper's Fig. 3 convention.
C_ANALYTICAL, C_DSM, C_GAUSS = "tab:red", "tab:blue", "tab:green"

# Paper Appendix A.2 defines TWO score-training configurations, and they are
# not interchangeable:
#
#   "attribution" - used for the D&A sweep of Fig. 1. 2000 epochs of the DSM
#       loss ALONE (no Stein penalty during training), followed by an affine
#       calibration enforcing Eq. (80): <s(X)>_0 = 0 and <(X-mu1)s(X)>_0 = -1.
#   "kernels"     - used for the state-dependent kernels and reconstruction of
#       Sec. 3.3. 3000 epochs with the ramped polynomial Stein penalty through
#       K_S = 4, then a final degree-four correction.
#
# lambda_max = 0 switches the Stein penalty off without touching the training
# module: d.train still evaluates the residuals (useful as a diagnostic) but
# multiplies them by zero, so the objective is pure DSM.
CONFIG = "attribution"

# Cached weights, and the flag that reuses them instead of retraining.
CKPT = "three_panel_model.pt"
REPLOT = "--replot" in sys.argv

CONFIGS = {
    "attribution": {"train": {"epochs": 2000, "lambda_max": 0.0}, "calib_degree": 1},
    "kernels": {"train": {"epochs": 3000, "lambda_max": 0.7}, "calib_degree": 4},
}


def analytical_score(x):
    return (2 / d.sigma**2) * (F + a * x + b * x**2 - c * x**3)


def analytical_log_density(x):
    V = F * x + a * x**2 / 2 + b * x**3 / 3 - c * x**4 / 4
    return (2 / d.sigma**2) * V


def analytical_moments():
    """Exact stationary mean/std of rho, by quadrature.

    The closed-form OU expressions do not apply once c != 0, so the moments
    of rho ~ exp((2/sigma^2) V(x)) are integrated numerically instead.
    """
    x = np.linspace(-4.0, 4.0, 200_001)
    rho = normalize_density(x, analytical_log_density(x))
    mu = _trapz(x * rho, x)
    var = _trapz((x - mu) ** 2 * rho, x)
    return mu, np.sqrt(var)


def replicate_seed(r, seed_seq=None):
    """The seed for replicate r: a pure function of the root seed and r.

    Derived by hand rather than with seed_seq.spawn(n_rep) because spawn()
    mutates its parent, so what it returns depends on how many children were
    requested and how many times it has been called. Here replicate 7 is
    always the same trajectory whether the ensemble has 8 members or 800, so
    you can grow N_REPLICATES, or rerun one suspicious replicate on its own,
    without perturbing any of the others.
    """
    seed_seq = SS_ENSEMBLE if seed_seq is None else seed_seq
    return np.random.SeedSequence(
        entropy=seed_seq.entropy, spawn_key=seed_seq.spawn_key + (r,)
    )


def generate_ensemble(n_steps=N_REF, n_rep=N_REPLICATES, dt=DT, thin=THIN,
                      seed_seq=None):
    """Euler-Maruyama for dx = F(x) dt + sigma dW, vectorized over replicates.

    Returns an (n_steps // thin, n_rep) float32 array holding every thin-th
    state. Each column is an independent trajectory following Appendix A.1
    exactly: X0 = 0, no burn-in. Advancing all replicates in lockstep makes
    the per-step cost one length-n_rep numpy op, so 256 replicates cost less
    than twice what 24 cost - the loop overhead, not the width, dominates.

    Thinning is a memory measure, not a statistical one. Every retained state
    is still a draw from the same chain, and dropping the states in between
    removes no information about rho because they were nearly determined by
    their neighbours anyway. What it buys is the ability to run many more
    replicates, which IS the thing that adds information.

    Each replicate draws from its own stream (see replicate_seed), in blocks
    of NOISE_BLOCK steps so peak memory stays bounded. Drawing one length-
    n_rep vector per timestep from a single shared stream would be simpler,
    but would make every trajectory depend on n_rep - the reproducibility
    trap this function exists to avoid.

    Equivalent to d.generate_scalar_data for n_rep = 1, thin = 1; kept here
    rather than added to dsm_with_stein so the training module is untouched.
    """
    rngs = [np.random.default_rng(replicate_seed(r, seed_seq)) for r in range(n_rep)]

    # float64 deliberately. Held as float32 this array would halve in size,
    # but then `noise * dw[i]` is a Python float times a float32 array, whose
    # result dtype depends on numpy's casting rules: float32 under numpy 1.x
    # value-based casting, and float32 for a length-n row but float64 for a
    # single scalar - so a replicate's path would silently change with n_rep
    # and with the numpy version. Keeping the increments in float64 makes the
    # integration bit-identical across both.
    block = min(NOISE_BLOCK, n_steps)
    dw = np.empty((block, n_rep), dtype=np.float64)

    x = np.zeros(n_rep)
    out = np.empty((n_steps // thin, n_rep), dtype=np.float32)
    noise = d.sigma * np.sqrt(dt)
    kept = 0
    for start in range(0, n_steps, block):
        span = min(block, n_steps - start)
        for r in range(n_rep):
            dw[:span, r] = rngs[r].standard_normal(span)
        for i in range(span):
            drift = F + a * x + b * x**2 - c * x**3
            x = x + drift * dt + noise * dw[i]
            if (start + i) % thin == thin - 1:
                out[kept] = x
                kept += 1
    return out[:kept]


def report_symmetry(traj, thin=THIN):
    """Print the symmetry diagnostics that the ensemble is meant to fix."""
    mu1 = traj.mean(axis=0)
    p_right = (traj > 0).mean(axis=0)
    n_rep = traj.shape[1]
    n_steps = traj.shape[0] * thin
    n_eff = n_rep * n_steps / TAU_STEPS
    print(
        f"ensemble: {n_rep} x {n_steps:,} steps = "
        f"{n_rep * n_steps * DT:,.0f} time units; "
        f"{traj.size:,} samples retained (thin={thin})"
    )
    print(
        f"  N_eff ~ {n_eff:,.0f} independent samples "
        f"(tau_int = {TAU_STEPS * DT:.0f} time units)"
    )
    print(
        f"  per-trajectory mu_1  : mean {mu1.mean():+.4f}  "
        f"spread {mu1.std(ddof=1):.4f} (exact 0)"
    )
    print(
        f"  per-trajectory P(x>0): mean {p_right.mean():.4f}  "
        f"spread {p_right.std(ddof=1):.4f} (exact 0.5)"
    )
    print(
        f"  pooled mu_1 = {traj.mean():+.5f} | pooled P(x>0) = "
        f"{(traj > 0).mean():.5f}"
    )
    # Returned so the sidecar records the error bars, not just the estimates:
    # a replication run gets to check its numbers against this spread.
    return {
        "n_replicates": int(n_rep),
        "n_steps": int(traj.shape[0]),
        "mu_1_per_replicate": mu1.tolist(),
        "p_right_per_replicate": p_right.tolist(),
        "mu_1_spread": float(mu1.std(ddof=1)),
        "p_right_spread": float(p_right.std(ddof=1)),
        "mu_1_pooled": float(traj.mean()),
        "p_right_pooled": float((traj > 0).mean()),
    }


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def write_metadata(path, diagnostics, extra):
    """Record everything needed to reproduce this figure, next to the figure.

    Seeds alone are not enough: the same seed on a different numpy, or against
    a different version of dsm_with_stein, gives different numbers. So the
    sidecar pins library versions and hashes both source files. The training
    hyperparameters are read out of d.train's signature rather than copied,
    so they cannot drift out of sync with the module actually being run.
    """
    train_args = {
        k: v.default
        for k, v in inspect.signature(d.train).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    meta = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root_seed_entropy": ROOT_SS.entropy,
        "substreams": ["ensemble", "training_subset", "torch"],
        "system": {"F": F, "a": a, "b": b, "c": c, "sigma": d.sigma,
                   "potential": "U(x) = x^4/4 - x^2/2"},
        "sampling": {"dt": DT, "n_steps": N_REF, "n_replicates": N_REPLICATES,
                     "x0": 0.0, "burn_in": 0, "thin": THIN,
                     "tau_steps": TAU_STEPS},
        "training": train_args,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__,
                     "platform": platform.platform()},
        "source_sha256": {
            "three_panel_figure.py": _sha256(__file__),
            "dsm_with_stein.py": _sha256(d.__file__),
        },
        "diagnostics": diagnostics,
        **extra,
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved {path}")


def score_on(model, x, batch=1 << 16):
    """Evaluate the score network on a numpy array, in batches."""
    out = np.empty(x.size, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, x.size, batch):
            chunk = torch.tensor(x[i:i + batch], dtype=torch.float32).view(-1, 1)
            out[i:i + batch] = model(chunk).numpy().ravel()
    return out


def stein_basis(x, mu, sigma_x, degree):
    """Test functions phi_k(x) = z^k, z = (x - mu)/sigma_x, and phi_k'(x).

    Standardizing is a conditioning choice, not a modelling one: scaling a
    test function rescales its Stein residual but not the constraint, and
    span{1, z} = span{1, x - mu}, so the calibrated score is identical to
    using the raw (x - mu1) basis of Eq. (80) - just better conditioned.
    """
    z = (x - mu) / sigma_x
    phi = np.stack([z**k for k in range(degree + 1)])
    dphi = np.stack([
        (k / sigma_x) * z ** (k - 1) if k else np.zeros_like(z)
        for k in range(degree + 1)
    ])
    return phi, dphi


def stein_residuals(phi, dphi, s):
    """r_k = <phi_k' + phi_k s>, the empirical Stein residual of Eq. (16)."""
    return dphi.mean(axis=1) + (phi * s).mean(axis=1)


def stein_calibration(model, x, mu, sigma_x, degree):
    """Post-training projection onto the Stein constraint set, Eqs. (19)-(21).

    Returns (s_cal, kappa, r_before, r_after) where s_cal(x_new) applies the
    correction s_theta(x) + sum_j kappa_j phi_j(x) at new points. Solved by
    least squares rather than an explicit inverse: the Gram matrix of raw
    polynomial moments is badly conditioned once degree reaches 4, since it
    involves standardized moments up to order 2*degree.
    """
    phi, dphi = stein_basis(x, mu, sigma_x, degree)
    s = score_on(model, x)
    r_before = stein_residuals(phi, dphi, s)

    G = phi @ phi.T / x.size
    b = -r_before
    kappa, *_ = np.linalg.lstsq(G, b, rcond=None)

    r_after = stein_residuals(phi, dphi, s + kappa @ phi)

    def s_cal(x_new):
        phi_new, _ = stein_basis(x_new, mu, sigma_x, degree)
        return score_on(model, x_new) + kappa @ phi_new

    return s_cal, kappa, r_before, r_after


def gaussian_score(x, mu, sigma_x):
    return -(x - mu) / sigma_x**2


def normalize_density(x_grid, unnorm_log_density):
    log_d = unnorm_log_density - unnorm_log_density.max()
    density = np.exp(log_d)
    area = _trapz(density, x_grid)
    return density / area


def main():
    # Seeds torch's weight init and minibatch order from the same root as the
    # data, so a rerun reproduces the figure and not merely the ensemble.
    # (Reductions on CPU are not bitwise-portable across thread counts and
    # BLAS builds, so expect agreement to a few ULP, not exact equality.)
    torch.manual_seed(int(SS_TORCH.generate_state(1, dtype=np.uint32)[0]))

    print(f"integrating reference ensemble (root seed {ROOT_SS.entropy})...")
    traj = generate_ensemble()
    diagnostics = report_symmetry(traj)

    # Everything downstream of the figure uses the pooled ensemble.
    x_np = traj.reshape(-1)

    # Train on the whole thinned ensemble. It is the same size as the paper's
    # single-trajectory budget, so d.train costs the same per epoch, but it
    # carries ~10x the effective sample size: 256 independent chains rather
    # than one. Subsetting further would only discard information, and
    # feeding the un-thinned 128M states instead would add ~17 min/run of
    # torch.randperm overhead for samples that are nearly duplicates.
    data = torch.tensor(x_np, dtype=torch.float32).view(-1, 1)
    print(f"training on {data.numel():,} samples")

    cfg = CONFIGS[CONFIG]
    # Training is ~28 min and the figure gets restyled far more often than the
    # score changes, so the trained weights are cached. `--replot` reuses them
    # and re-does only the (16 s) sampling and the plotting. The cache records
    # the config it was trained under and is refused if that no longer matches.
    if REPLOT and os.path.exists(CKPT):
        ck = torch.load(CKPT)
        if ck["config"] != CONFIG:
            raise SystemExit(
                f"{CKPT} was trained with config '{ck['config']}', not '{CONFIG}'. "
                f"Re-run without --replot."
            )
        model = d.ScoreNet(size=1024)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        final_loss = ck["final_loss"]
        print(f"loaded cached model from {CKPT} (final loss={final_loss:.4f})")
    else:
        print(f"training, config '{CONFIG}': {cfg['train']}")
        model, final_loss = d.train(data, **cfg["train"])
        print(f"training done, final loss={final_loss:.4f}")
        torch.save(
            {"state_dict": model.state_dict(), "final_loss": float(final_loss),
             "config": CONFIG, "root_seed_entropy": ROOT_SS.entropy},
            CKPT,
        )
        print(f"Saved {CKPT}")

    mu = float(x_np.mean())
    sigma_x = float(x_np.std(ddof=1))

    # Paper Eqs. (19)-(21). For CONFIG='attribution' this is the affine
    # calibration of Eq. (80), the score the D&A sweep of Fig. 1 actually uses.
    s_cal, kappa, r_before, r_after = stein_calibration(
        model, x_np, mu, sigma_x, cfg["calib_degree"]
    )
    print(f"Stein calibration, degree {cfg['calib_degree']}: kappa = "
          f"{np.array2string(kappa, precision=4)}")
    print(f"  residuals before: {np.array2string(r_before, precision=4)}")
    print(f"  residuals after : {np.array2string(r_after, precision=4)}")

    mu_exact, sigma_exact = analytical_moments()
    print(
        f"sampled mean/std = {mu:.4f}/{sigma_x:.4f} | "
        f"exact mean/std = {mu_exact:.4f}/{sigma_exact:.4f}"
    )

    # Trim to the populated region: the extreme order statistics are single
    # draws, where the learned score is unconstrained.
    x_max = float(np.quantile(np.abs(x_np), 0.9995))
    x_grid = np.linspace(-x_max, x_max, 601)

    s_analytical = analytical_score(x_grid)
    s_gaussian = gaussian_score(x_grid, mu, sigma_x)
    # The calibrated score is the one D&A uses, so it is the one plotted.
    s_dsm = s_cal(x_grid)

    log_rho_analytical = analytical_log_density(x_grid)
    pdf_analytical = normalize_density(x_grid, log_rho_analytical)

    pdf_gaussian = (1 / np.sqrt(2 * np.pi * sigma_x**2)) * np.exp(
        -((x_grid - mu) ** 2) / (2 * sigma_x**2)
    )

    log_rho_dsm = cumulative_trapezoid(s_dsm, x_grid, initial=0)
    pdf_dsm = normalize_density(x_grid, log_rho_dsm)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # The pooled ensemble is thinned, so its rows are 12.5 time units apart
    # and would plot as noise. Re-integrate replicate 0 unthinned instead:
    # same seed stream, so this is exactly the opening of that trajectory.
    t = np.arange(N_TRAJ_PLOT) * DT
    traj0 = generate_ensemble(n_steps=N_TRAJ_PLOT, n_rep=1, thin=1)[:, 0]
    axes[0].plot(t, traj0, lw=0.7, color="black")
    for well in (-1.0, 1.0):
        axes[0].axhline(well, color="gray", lw=0.8, linestyle="--")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title(
        f"Representative trajectory (1 of {N_REPLICATES}, sigma={d.sigma})"
    )

    axes[1].plot(x_grid, s_analytical, label="Analytical", color=C_ANALYTICAL, lw=2.5)
    axes[1].plot(x_grid, s_dsm, label="DSM", color=C_DSM, linestyle="--", lw=1.8)
    axes[1].plot(x_grid, s_gaussian, label="Gaussian", color=C_GAUSS, linestyle=":", lw=1.8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("score s(x)")
    axes[1].set_title(
        f"Score function"
        # f"{cfg['train']['epochs']} ep, deg-{cfg['calib_degree']} calib.)"
    )
    axes[1].legend()

    axes[2].hist(
        x_np, bins=200, range=(-x_max, x_max), density=True,
        color="lightgray", label=f"Empirical ({N_REPLICATES} traj.)", zorder=0,
    )
    axes[2].plot(x_grid, pdf_analytical, label="Analytical", color=C_ANALYTICAL, lw=2.5)
    axes[2].plot(x_grid, pdf_dsm, label="DSM", color=C_DSM, linestyle="--", lw=1.8)
    axes[2].plot(x_grid, pdf_gaussian, label="Gaussian", color=C_GAUSS, linestyle=":", lw=1.8)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("rho(x)")
    axes[2].set_title("Stationary PDFs")
    axes[2].legend()

    fig.tight_layout()
    out = "three_panel_figure_final.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")

    write_metadata(
        out.replace(".png", ".json"),
        diagnostics,
        {
            "moments": {
                "mu_sampled": mu, "sigma_sampled": sigma_x,
                "mu_exact": float(mu_exact), "sigma_exact": float(sigma_exact),
            },
            "final_loss": float(final_loss),
            "score_config": {
                "name": CONFIG,
                **cfg["train"],
                "calib_degree": cfg["calib_degree"],
                "kappa": kappa.tolist(),
                "stein_residuals_before": r_before.tolist(),
                "stein_residuals_after": r_after.tolist(),
            },
        },
    )


if __name__ == "__main__":
    main()
