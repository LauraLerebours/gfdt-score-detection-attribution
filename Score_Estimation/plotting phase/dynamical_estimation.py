import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
start_time = time.time()


F, a, b, c = 0.0, 1.0, 0.0, 1.0  
sigma = 1.0                      
dt = 0.01                        
num_steps = 500000      
num_steps_R = 1000         
######## FIRST MOMENT #########
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

eta = 0.05  
base_x = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=0.0, seed=123) #Used as true pdf for sampling initial conditions

initial_conditions = rand_start.randint(low=0, high=len(base_x)-1, size=400)  # Generate 200 random initial conditions
trajectory = []
unperturbed = []
perturbed = []
# Simulate trajectories for each initial condition and compute the average difference at the final time step
def simulate_and_compute_diff(initial, eta, dt, num_steps, seed):
    i = base_x[initial] # Get the state variable value at the initial condition index
    # print(i)
    x_0 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=i, seed=seed)
    x_1 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, x_0=eta + i, seed=seed)
    unperturbed.append(x_0)
    perturbed.append(x_1)
    diff = [x_1[j] - x_0[j] for j in range(len(x_0))]
    print(diff)

    return diff 
# n = [n for n in range(1, 50)]
rand_seed = np.random.RandomState(seed=456)  # For reproducibility of random seeds for each n

differences = []
i = 0
for initial in initial_conditions:
    print(i)
    i += 1
    seed_call = rand_seed.randint(low=0, high=10000)  # Generate a random seed for this initial condition 

    # print(initial)
    diff = simulate_and_compute_diff(initial, eta, dt, num_steps_R, seed_call)
    differences.append(diff)  # Append the difference at the final time step
    # print(f"Initial condition {initial}: Difference = {diff}")
    # time.sleep(3)  # Sleep to avoid overwhelming the system with print statements
    
for i in range(len(differences[0])):  # Loop over time steps
    avg_diff = np.mean([diff[i] for diff in differences])  # Average over all initial conditions
    trajectory.append(avg_diff)  # Append the average difference for this time step



end_time = time.time()
print(f"Total simulation time: {end_time - start_time:.2f} seconds")

plt.plot(trajectory, label='Average Trajectory Difference', color='red')
plt.title('Dynamical Estimation of Perturbation Effect')
plt.xlabel('Time Steps')
plt.ylabel('State Variable x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


########## SECOND MOMENT ##########
mu = np.mean(base_x)  # Compute the mean of the base_x trajectory
trajectory2 = []
# unperturbed_trajectories = []
# perturbed_trajectories = []
print("Simulating for second moment estimation...")
unperturbed_trajectories = []
perturbed_trajectories = []
unperturbed_trajectories = [u - mu for u in unperturbed]
perturbed_trajectories = [p - mu for p in perturbed]
# for initial in initial_conditions:
#     seed_call = rand_seed.randint(low=0, high=10000)
#     i = base_x[initial] # Get the state variable value at the initial condition index
#     x_0 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, i, seed_call)
#     x_1 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, eta + i, seed_call)
#     unperturbed_trajectories.append([x**2 for x in x_0])
#     perturbed_trajectories.append([x**2 for x in x_1])
# j = 0
# def second_moment_diff(initial):
#     global j
  
#     seed_call = rand_seed.randint(low=0, high=10000)
#     i = base_x[initial] # Get the state variable value at the initial condition index
#     x_0 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, i, seed_call)
#     x_1 = simulate_time_series(F, a, b, c, sigma, dt, num_steps, eta + i, seed_call)
#     unperturbed_trajectories.append([(x - mu)**2 for x in x_0])
#     perturbed_trajectories.append([(x - mu)**2 for x in x_1])
#     print(f"Completed simulation for initial condition {initial}")
#     j += 1
#     print(j)

# with ThreadPoolExecutor() as executor:
#     executor.map(lambda initial: second_moment_diff(initial), initial_conditions)

# for i in range(len(unperturbed_trajectories[0])):  # Loop over time steps
#     avg_unperturbed = np.mean([traj[i] for traj in unperturbed_trajectories])  # Average over all initial conditions
#     avg_perturbed = np.mean([traj[i] for traj in perturbed_trajectories])  # Average over all initial conditions
#     trajectory2.append(avg_perturbed - avg_unperturbed)  # Append the difference of averages for this time step

plt.plot(trajectory2, label='Difference of Average Squared Trajectories', color='blue')
plt.title('Dynamical Estimation of Perturbation Effect (Second Moment)')
plt.xlabel('Time Steps')
plt.ylabel('State Variable x^2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Simulate trajectories for each initial condition and compute the average difference between perturbed and unperturbed trajectories

# for n  in range(1, 50):
#     print(f'Simulating for n={n} time steps...')
#     avg = []
#     for initial in initial_conditions:
#         i = base_x[initial]
#         x_0 = simulate_time_series(F, a, b, c, sigma, dt, int(n/dt), x_0=i)
#         x_1 = simulate_time_series(F, a, b, c, eta, dt, int(n/dt), x_0=eta + i)
#         diff = x_0[-1] - x_1[-1]
#         avg.append(diff)
#     mean_diff = np.mean(avg)
#     trajectory.append(mean_diff)

# def sim_n(n):
#     with ThreadPoolExecutor() as executor:
#         print(f'Simulating for n={n} time steps...')
#         num_steps = int(n / dt)
#         diffs = list(executor.map(lambda initial: simulate_and_compute_diff(initial, eta, dt, num_steps_R, seed=initial), initial_conditions))
#         mean_diff = np.mean(diffs)
#         return mean_diff

# with ThreadPoolExecutor() as executor2:
#     trajectory = list(executor2.map(lambda val: sim_n(val), n))