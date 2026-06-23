import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Define Model Parameters 
# (Placeholder values; the authors' exact parameters are in SI Appendix, Table S1)
F = .6                             # Constant force
a = -0.0222                      # Linear coefficient
b = -0.2                            # Quadratic coefficient
c = 0.0494                            # Cubic coefficient
sigma = .7071                     # Noise amplitude
dt = 0.01                        # Time step
num_steps = 500000               # Number of simulation steps

# 2. Simulate Unperturbed Time-Series (Euler-Maruyama integration)
x = np.zeros(num_steps)
for i in range(num_steps - 1):
    drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
    diffusion = sigma * np.sqrt(dt) * np.random.randn()
    x[i+1] = x[i] + drift * dt + diffusion

# 3. Calculate Conjugate Observable B(x) = -s(x)
# The analytical score function s(x) = (2 / sigma^2) * (F + a*x + b*x^2 - c*x^3)
s_x = (2.0 / sigma**2) * (F + a*x + b*(x**2) - c*(x**3))
B_x = -s_x

# 4. Calculate Response Function R(t) = < A(x(t)) B(x(0)) >_0
# For the 1st moment response, the observable A(x) = x
A_x = x

max_lag_steps = 5000  # Calculate response up to 500 time steps (t = 5.0)
R_t = np.zeros(max_lag_steps)

# Compute time-lagged cross-correlation (the expectation value <.>_0)
for lag in range(max_lag_steps):
    # We multiply A(x) at time t by B(x) at time 0, and take the mean over the ensemble
    R_t[lag] = np.mean(A_x[lag:] * B_x[:num_steps-lag])

# 5. Plot the Response Function
time_lags = np.arange(max_lag_steps) * dt

plt.figure(figsize=(6, 4))
plt.plot(time_lags, R_t, label='1st Moment Response', color='blue', linewidth=2)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('1st Moment Response Function via GFDT')
plt.xlabel('Time lag')
plt.ylabel('Response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

########PDF#########
# 2. Simulate Unperturbed Time-Series (to represent empirical data)
x_sim = np.zeros(num_steps)
for i in range(num_steps - 1):
    drift = F + a*x_sim[i] + b*(x_sim[i]**2) - c*(x_sim[i]**3)
    diffusion = sigma * np.sqrt(dt) * np.random.randn()
    x_sim[i+1] = x_sim[i] + drift * dt + diffusion

# 3. Calculate the Analytical PDF
# Define the unnormalized distribution
def unnormalized_pdf(x):
    return np.exp((2.0 / sigma**2) * (F*x + (a/2.0)*x**2 + (b/3.0)*x**3 - (c/4.0)*x**4))

# Calculate the normalization constant N by integrating from -infinity to +infinity
N, _ = quad(unnormalized_pdf, -np.inf, np.inf)

# Create x values for plotting the smooth lines
x_vals = np.linspace(-4, 4, 400)
# Divide by N to get the true probability density
analytical_pdf = unnormalized_pdf(x_vals) / N

# 4. Calculate the Gaussian Approximation PDF
# A normal distribution using the mean and variance of the simulated data
mu = np.mean(x_sim)
variance = np.var(x_sim)
gaussian_pdf = (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x_vals - mu)**2 / variance))

# 5. Plotting all three together (Recreating Figure 1, leftmost column)
plt.figure(figsize=(6, 5))

# Plot Analytical "Ground Truth" (Blue line)
plt.plot(x_vals, analytical_pdf, label='Analytical PDF (True)', color='blue', linewidth=2)

# Plot Gaussian Approximation (Black line)
plt.plot(x_vals, gaussian_pdf, label='Gaussian Approximation', color='black', linewidth=1.5)

# Plot Empirical PDF from our simulation (Red dashed/histogram)
# In the paper, this would be the output of integrating the Langevin dynamics with the KGMM score
plt.hist(x_sim, bins=100, density=True, color='red', alpha=0.3, label='Empirical / KGMM')

plt.title('Unperturbed Steady-State PDF')
plt.xlabel('x Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-4, 4])
plt.tight_layout()
plt.show()
# ###### all the moments together ######
max_lag_steps = 5000
time_lags = np.arange(max_lag_steps) * dt

# Calculate the mean of the unperturbed simulation
mu = np.mean(x)

# Define the observables A(x) for each central moment
A_x_1 = x                 # 1st moment (Mean)
A_x_2 = (x - mu)**2       # 2nd central moment (Variance)
A_x_3 = (x - mu)**3       # 3rd central moment (Skewness)
A_x_4 = (x - mu)**4       # 4th central moment (Kurtosis)

# Initialize arrays to store the responses
R_t_1 = np.zeros(max_lag_steps)
R_t_2 = np.zeros(max_lag_steps)
R_t_3 = np.zeros(max_lag_steps)
R_t_4 = np.zeros(max_lag_steps)

# Compute time-lagged cross-correlation for all moments simultaneously
for lag in range(max_lag_steps):
    R_t_1[lag] = np.mean(A_x_1[lag:] * B_x[:num_steps-lag])
    R_t_2[lag] = np.mean(A_x_2[lag:] * B_x[:num_steps-lag])
    R_t_3[lag] = np.mean(A_x_3[lag:] * B_x[:num_steps-lag])
    R_t_4[lag] = np.mean(A_x_4[lag:] * B_x[:num_steps-lag])

# 5. Plot the Response Functions (Recreating the right side of Figure 1)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))

moments = [R_t_1, R_t_2, R_t_3, R_t_4]
titles = ['1st moment', '2nd moment', '3rd moment', '4th moment']

for i in range(4):
    axs[i].plot(time_lags, moments[i], label='Analytical GFDT Response', color='blue')
    axs[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Time lag')
    if i == 0:
        axs[i].set_ylabel('Response')

plt.tight_layout()
plt.show()