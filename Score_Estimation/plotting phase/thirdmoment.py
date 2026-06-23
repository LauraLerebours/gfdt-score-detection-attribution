import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.01                        
num_steps = 500000      
num_steps_R = 3000         
######## SECOND MOMENT #########
 # For reproducibility
rand_start = np.random.RandomState(seed=123)  # For reproducibility of initial conditions  
# function to simulate time series with given initial condition 
def simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0, seed):
    rand_noise = np.random.RandomState(seed=seed) 
    x = np.zeros(num_steps)
    x[0] = x_0
    for i in range(num_steps - 1):
        drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
        diffusion = sigma * np.sqrt(dt) * rand_noise.randn()
        x[i+1] = x[i] + drift * dt + diffusion
    return x
x_long = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=0.0, seed=123)
burn = 50000
mu = np.mean(x_long[burn:])
base_x = x_long[burn:]
# mu = np.mean(simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=0.0, seed=123)) #Used as true mean for sampling initial conditions
# function to simulate time series and square with given initial condition
def simulate_time_series_square(F, a, b, c, sigma, dt, num_steps, x_0, seed):
    rand_noise = np.random.RandomState(seed=seed) 
    x = np.zeros(num_steps)
    square = []
    x[0] = x_0
    square.append((x[0] - mu)**3)
    for i in range(num_steps - 1):
        drift = F + a*x[i] + b*(x[i]**2) - c*(x[i]**3)
        diffusion = sigma * np.sqrt(dt) * rand_noise.randn()
        x[i+1] = x[i] + drift * dt + diffusion
        square.append((x[i+1] - mu)**3)
    return x, square
# pertubation magnitude
eta = 0.2  
base_x, base_square = simulate_time_series_square(F, a, b, c, sigma, dt, num_steps, 0.0, 123) #Used as true pdf for sampling initial conditions
initial_conditions = rand_start.randint(low=0, high=len(base_x)-1, size=1000)  # Generate 400 random initial conditions
trajectory = []
differences = []
# Simulate trajectories for each initial condition and compute the average difference at the final time step
def simulate_and_compute_diff(initial, eta, dt, num_steps, seed):
    x_init = base_x[initial] # Get the state variable value at the initial condition index'
    _,x_0_2 = simulate_time_series_square(F, a, b, c, sigma, dt, num_steps, x_init, seed)
    _,x_1_2 = simulate_time_series_square(F, a, b, c, sigma, dt, num_steps, eta + x_init, seed)
    diff = [(x_1_2[j] - x_0_2[j]) for j in range(len(x_0_2))]
    return diff

# Generate data for each initial condition
rand_seed = np.random.RandomState(seed=456)  # For reproducibility of random seeds for each n
j = 0
for initial in initial_conditions:
    print(j)
    j += 1
    seed_call = rand_seed.randint(low=0, high=10000)  # Generate a random seed for this initial condition 
    diff = simulate_and_compute_diff(initial, eta, dt, num_steps_R, seed_call) # Call the function to simulate and compute the difference
    differences.append(diff) # Store the difference for this initial condition

# get the averages and subtract them
trajectory = [np.mean([differences[j][i] for j in range(len(differences))]) for i in range(num_steps_R)] # Compute the average difference at each time step
# window = 100
# trajectory_smooth = np.convolve(
#     trajectory,
#     np.ones(window) / window,
#     mode="same"
# )
# Plot everything
end_time = time.time()
print(f"Total simulation time: {end_time - start_time:.2f} seconds")
time_axis = np.arange(num_steps_R) * dt
plt.plot(time_axis, trajectory, label='Average Cubed Trajectory Difference', color='red')
plt.title('Dynamical Estimation of Perturbation Effect')
plt.xlabel('Time Steps')
plt.ylabel('Third moment response')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
