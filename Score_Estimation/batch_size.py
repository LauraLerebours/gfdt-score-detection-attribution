import dsm_with_stein
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import threading

batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
losses = [None] * len(batch_sizes)
threads = []

data = dsm_with_stein.generate_scalar_data(0.0, 1.0, 0.0, 1.0, 0.5, 0.01, 100000)
# use threading to speed up training for different batch sizes


# for batch_size in batch_sizes:
#     # Assuming you have a function to train the model with a specific batch size
#     model, perm, loss = dsm_with_stein.train(data, batch_size=batch_size)
#     losses[batch_sizes.index(batch_size)] = loss

from concurrent.futures import ProcessPoolExecutor


def run_one_batch_size(batch_size):
    print(f"Training with batch size: {batch_size}")
    model, perm, loss = dsm_with_stein.train(data, batch_size=batch_size)
    return batch_size, loss


# with ProcessPoolExecutor() as executor:
#     # for batch_size in batch_sizes:
#     #     print(f"Training with batch size: {batch_size}")
#     #     ls = [batch_size for _ in range(5)]  # Run each batch size 5 times for averaging
#     #     temp = list(executor.map(run_one_batch_size, ls) )
#     #     results = np.mean(temp, axis=0)
#     ls = [16 for _ in range(5) ]  # Run each batch size 5 times for averaging
#     results = list(executor.map(run_one_batch_size, ls))

losses = []

for i in range(10):  # Run batch size 10 times for averaging
    print(f"Run {i + 1}/10")
    _, loss = run_one_batch_size(2048)  # Change the batch size here
    losses.append(loss)


print("Results:", losses)
print("Averaged Results:", np.mean(losses))
print("Standard Deviation:", np.std(losses))


# for batch_size, loss in results:
#     i = batch_sizes.index(batch_size)
# #     losses[i] = loss

# # Plotting the results
# plt.plot(batch_sizes, losses, marker='o')
# plt.xlabel('Batch Size')
# plt.ylabel('Loss')
# plt.title('Loss vs Batch Size')
# plt.xscale('log', base=2)
# plt.show()
