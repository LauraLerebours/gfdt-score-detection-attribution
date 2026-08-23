import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCORE_DIRECTORY = Path(__file__).resolve().parents[1] / "Score_Estimation"
if str(SCORE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCORE_DIRECTORY))
import dsm_with_stein

from control_covariance import estimate_control_covariance

s = 0.5
dt = 0.05
N_real = 500
n_ens = 1000
n_steps = 100
n_steps_kernels = 300
eps = 0.06
eps_vals = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

np.random.seed(42)
torch.manual_seed(42)

def drift(x):
    F, a, b, c = 0.0, 1.0, 0.0, 1.0
    return  F + a*x + b*x**2 - c*x**3

#long trajectory and analytical score
N_long = 500_000
x = 0.0
traj = np.zeros(N_long)
score_exact = np.zeros(N_long)
for i in range(N_long):
    x += (drift(x))*dt + s*np.sqrt(dt)*np.random.randn()
    traj[i] = x
    score_exact[i] = 2*(drift(x)) / s**2

#multiple means for Y_a(t)
mu1 = traj.mean()
mu2 = ((traj-mu1)**2).mean()
mu3 = ((traj-mu1)**3).mean()

score_gauss = -(traj - mu1)/mu2

m2 = (traj**2).mean()   #raw second moment for conjugate observables

traj_thin = traj[::200]

# learned score
def stein_calibrate_score(x, score):
    z = x - x.mean()
    # enforce <score> = 0
    score = score - score.mean()
    # enforce <(X-mu) score> = -1
    score = score * (-1.0 / np.mean(z * score))

    return score

data_tensor = torch.tensor(traj, dtype=torch.float32).view(-1, 1)
score_model, _ = dsm_with_stein.train(data_tensor, lambda_max=0.7, epochs=3000, batch_size=1024, K=4)
score_model.eval()
with torch.no_grad():
    score_dsm = score_model(data_tensor).numpy().flatten()
score_dsm = stein_calibrate_score(traj, score_dsm)
score_exact = stein_calibrate_score(traj, score_exact)
score_gauss = stein_calibrate_score(traj, score_gauss)

#state-dependent conjugate observables
traj_centered = traj - mu1
B_exact = [-1.0 - traj*score_exact, -2.0*traj - (traj**2 - m2)*score_exact]
B_dsm   = [-1.0 - traj*score_dsm, -2.0*traj - (traj**2 - m2)*score_dsm]
B_gauss = [-1.0 - traj*score_gauss, -2.0*traj - (traj**2 - m2)*score_gauss]

#state_independent conjugate observables
# B_exact = -score_exact
# B_dsm   = -score_dsm
# B_gauss = -score_gauss

#kernels
observables = [traj_centered, traj_centered**2 - mu2, traj_centered**3 - mu3]
R_exact = np.zeros((3, 2, n_steps_kernels))
R_dsm   = np.zeros((3, 2, n_steps_kernels))
R_gauss = np.zeros((3, 2, n_steps_kernels))
for a in range(3):
    psi = observables[a]
    for p in range(2):
        for t in range(n_steps_kernels):
            R_exact[a, p, t] = np.mean(psi[t:] * B_exact[p][:N_long-t])
            R_dsm[a, p, t]   = np.mean(psi[t:] * B_dsm[p][:N_long-t])
            R_gauss[a, p, t] = np.mean(psi[t:] * B_gauss[p][:N_long-t])

#state_independent kernels
# R_exact = np.zeros((3, n_steps))
# R_dsm   = np.zeros((3, n_steps))
# R_gauss = np.zeros((3, n_steps))
# for a in range(3):
#     psi = observables[a]
#     for t in range(n_steps):
#         R_exact[a, t] = np.mean(psi[t:] * B_exact[:N_long-t])
#         R_dsm[a, t]   = np.mean(psi[t:] * B_dsm[:N_long-t])
#         R_gauss[a, t] = np.mean(psi[t:] * B_gauss[:N_long-t])

#kernel plot
# lag_axis = np.arange(n_steps_kernels) * dt

# fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
# for a in range(3):
#     for p in range(2):
#         ax[a, p].plot(lag_axis, R_exact[a, p], 'r', label='exact')
#         ax[a, p].plot(lag_axis, R_dsm[a, p],   'b', label='DSM')
#         ax[a, p].plot(lag_axis, R_gauss[a, p], 'g', label='Gaussian')
#         ax[a, p].axhline(0, color='gray', lw=0.5)
# ax[0, 0].set_title('$G_1 = x$'); ax[0, 1].set_title('$G_2 = x^2 - m_2$')
# for a, name in enumerate(['mean', 'variance', 'skewness']):
#     ax[a, 0].set_ylabel(name)
# ax[2, 0].set_xlabel('lag'); ax[2, 1].set_xlabel('lag')
# ax[0, 0].legend()
# plt.tight_layout(); plt.show()

# covariance of one complete observable time series from an unforced trajectory
d = 3 * n_steps
control_rng = np.random.default_rng(20260821)
control_indices = control_rng.choice(len(traj_thin), size=n_ens, replace=False)
C_hat, _, control_vectors = estimate_control_covariance(
    traj_thin[control_indices],
    drift=drift,
    observables=(
        lambda values: values - mu1,
        lambda values: (values - mu1) ** 2 - mu2,
        lambda values: (values - mu1) ** 3 - mu3,
    ),
    n_steps=n_steps,
    dt=dt,
    sigma=s,
    rng=control_rng,
    alpha=1.0e-2,
)
# C_hat / n_ens is the covariance of an ensemble mean.  The scalar n_ens
# cancels from GLS point estimates and is omitted from this weighting matrix.
C_inv = np.linalg.inv(C_hat)

#forcing functions
T = n_steps * dt
time_scale = np.arange(n_steps) * dt
eps1, eps2 = 0.06, 0.06
h1_true = eps1 * (1.0 + 0.25*np.sin(2*np.pi*time_scale/T))
h2_true = eps2 * (0.25 + 0.75*time_scale/T)
#state independent forcings
# g1 = np.ones(n_steps)
# g2 = time_scale / time_scale[-1]
# g_true = eps * (g1 + g2)      

Y_all = np.zeros((N_real, d))
for i in range(N_real):
    idx = np.random.randint(0, len(traj_thin) - n_ens)
    x0 = traj_thin[idx:idx+n_ens].copy()
    xu = x0.copy()
    xp = x0.copy()
    Y = np.zeros((3, n_steps))
    for t in range(n_steps):
        noise = s*np.sqrt(dt)*np.random.randn(n_ens)
        forcing = h1_true[t]*xp + h2_true[t]*(xp**2 - m2)   #state-dependent
        xu += (drift(xu))*dt + noise
        xp += (drift(xp) + forcing)*dt + noise
        Y[0, t] = xp.mean() - xu.mean() 
        Y[1, t] = ((xp-mu1)**2).mean() - ((xu-mu1)**2).mean()
        Y[2, t] = ((xp-mu1)**3).mean() - ((xu-mu1)**3).mean()
    Y_all[i] = np.concatenate([Y[0], Y[1], Y[2]])
Y = Y_all.mean(axis=0)

#block-Toeplitz operator
def build_operator(R):
    K = np.zeros((3*n_steps, n_steps))
    for a in range(3):                
        for n in range(n_steps):       
            for l in range(n+1):         
                K[a*n_steps + n, n - l] = dt * R[a, l]  
    return K

#state_independent operators 
# K_exact = build_operator(R_exact[:, :n_steps])
# K_dsm   = build_operator(R_dsm[:, :n_steps])
# K_gauss = build_operator(R_gauss[:, :n_steps])

K1_exact = build_operator(R_exact[:, 0, :])
K2_exact = build_operator(R_exact[:, 1, :])
K_exact  = np.hstack([K1_exact, K2_exact])      #(300, 200)

K1_dsm = build_operator(R_dsm[:, 0, :])
K2_dsm = build_operator(R_dsm[:, 1, :])
K_dsm  = np.hstack([K1_dsm, K2_dsm])

#discrete-time derivative operator
D_1 = np.zeros((n_steps, n_steps))          #first difference
for n in range(n_steps-1):
    D_1[n, n] = -1.0
    D_1[n, n+1] = 1.0

D_2 = np.zeros((n_steps, n_steps))          #second difference 
for n in range(n_steps-2):
    D_2[n, n] = 1.0
    D_2[n, n+1] = -2.0
    D_2[n, n+2] = 1.0

Z = np.zeros((n_steps, n_steps))
D1_blk = np.block([[D_1, Z], [Z, D_1]])     #(200, 200)
D2_blk = np.block([[D_2, Z], [Z, D_2]])

def deconvolve(K, Y, lam1, lam2):
    lhs = K.T @ C_inv @ K + lam1*(D1_blk.T @ D1_blk) + lam2*(D2_blk.T @ D2_blk)
    rhs = K.T @ C_inv @ Y
    return np.linalg.solve(lhs, rhs)

#state independent penalties
# def deconvolve(K, Y, lam1, lam2):
#     lhs = K.T @ C_inv @ K + lam1*(D_1.T @ D_1) + lam2*(D_2.T @ D_2)
#     rhs = K.T @ C_inv @ Y
#     return np.linalg.solve(lhs, rhs)

lam1, lam2 = 1.0, 1.0
g_exact = deconvolve(K_exact, Y, lam1, lam2)
g_dsm = deconvolve(K_dsm,   Y, lam1, lam2)

g1_exact, g2_exact = g_exact[:n_steps], g_exact[n_steps:]
g1_dsm, g2_dsm = g_dsm[:n_steps],   g_dsm[n_steps:]

#plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(time_scale, h1_true, 'k', label='true $h_1$')
ax[0].plot(time_scale, g1_exact, 'r--', label='exact')
ax[0].plot(time_scale, g1_dsm, 'b--', label='DSM')
ax[0].set_title('channel 1: $G_1=x$'); ax[0].set_xlabel('t'); ax[0].legend()
ax[1].plot(time_scale, h2_true, 'k', label='true $h_2$')
ax[1].plot(time_scale, g2_exact, 'r--', label='exact')
ax[1].plot(time_scale, g2_dsm, 'b--', label='DSM')
ax[1].set_title('channel 2: $G_2=x^2-m_2$'); ax[1].set_xlabel('t'); ax[1].legend()
plt.tight_layout(); plt.show()

#state independent plot
# plt.figure(figsize=(7, 4))
# plt.plot(time_scale, g_true,  'k',   label='true')
# plt.plot(time_scale, g_exact, 'r--', label='exact')
# plt.plot(time_scale, g_dsm,   'b--', label='DSM')
# plt.xlabel('t'); plt.ylabel('$g(t)$')
# plt.title('State-independent forcing recovery')
# plt.legend()
# plt.tight_layout()
# plt.savefig('recovery_state_independent.pdf', bbox_inches='tight')
# plt.show()
