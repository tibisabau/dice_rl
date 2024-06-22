# New file that calculates the MSE and plots the average per-step reward 
# and MSE vs KL divergence values

import matplotlib.pyplot as plt
import pickle
import numpy as np

def average(left, right, arr):
    avg = []
    for j in range(len(arr[0]['running_estimates'])):
        sum = 0
        for i in range (left, right):
            sum += arr[i]['running_estimates'][j] 
        avg.append(sum / 5)
    return avg

def avg_true(arr):
    avg = []
    for j in range(len(arr[0]['true'])):
        sum = 0
        for i in range (0, 5):
            sum += arr[i]['true'][j] 
        avg.append(sum / 5)
    return avg

with open('kl_values.pkl', 'rb') as f:
    kl = pickle.load(f)
kl_values = kl['kl_values']
with open('per_steps_values.pkl', 'rb') as f:
    ps = pickle.load(f)
ps_values = ps['dicts']
uniform_avg = average(0, 5, ps_values)
distance_avg = average(5, 10, ps_values)
mixed_avg = average(10, 15, ps_values)
close_avg = average(15, 20, ps_values)
fixed_avg = average(20, 25, ps_values)
far_avg = average(25, 30, ps_values)
edge_avg = average(30, 35, ps_values)
true = avg_true(ps_values)
true_arr = []
for i in range (0, 5):
    true_arr.append(ps_values[i]['true'][-1])

plt.figure(figsize=(12, 7))
plt.plot(ps_values[0]['steps'], uniform_avg, label=f'running estimates uniform kl={round(kl_values[0], 2)}')
plt.plot(ps_values[2]['steps'], mixed_avg, label=f'running estimates mixed mode kl={round(kl_values[2], 2)}')
plt.plot(ps_values[6]['steps'], edge_avg, label=f'running estimate edge bias kl={round(kl_values[6], 2)}')
plt.plot(ps_values[1]['steps'], distance_avg, label=f'running estimates distance based kl={round(kl_values[1], 2)}')
plt.plot(ps_values[5]['steps'], far_avg, label=f'running estimates remote kl={round(kl_values[5], 2)}')
plt.plot(ps_values[4]['steps'], fixed_avg, label=f'running estimates fixed point kl={round(kl_values[4], 2)}')
plt.plot(ps_values[3]['steps'], close_avg, label=f'running estimates target-centric kl={round(kl_values[3], 2)}')
plt.plot(ps_values[0]['steps'], true, label='True Value', linestyle='--', color='black')
plt.ylim(0.85, 1) 
plt.subplots_adjust(top=0.75)
plt.xlabel('Steps')
plt.ylabel('Cumulative Normalized Expected Reward')
plt.title('Running Estimates vs True Value Over Steps')
plt.legend(fontsize='large', bbox_to_anchor=(0.5, 1.4), loc='upper center', ncol=2)
plt.grid(True)
plt.savefig('plot_Every_Mismatch.png')
plt.show()

data = []
k = 0
for i in range(7):
    row = []
    for j in range (k, k + 5):
        row.append(ps_values[j]['running_estimates'][-1].numpy())
    data.append(row)
    k += 5
print(data)
mse_values = [np.mean((np.array(samples) - true_arr) ** 2) * 1000 for samples in data]
x_labels = np.around(kl_values, decimals=2)
x_strings = ["Uniform", "Distance Based", "Mixed Mode", "Target-Centric", "Fixed Point", "Remote", "Edge Bias"]
sorted_pairs = sorted(zip(x_labels, x_strings, data, mse_values))

x_labels, x_strings, data, mse_values = zip(*sorted_pairs)
x_labels = [f"{x_label}\n({secondary_label})" for x_label, secondary_label in zip(x_labels, x_strings)]
fig, ax = plt.subplots(figsize=(12, 7))

# Plotting the boxplots for each experiment
ax.boxplot(data, patch_artist=True)

# Annotate the MSE values
for i, mse in enumerate(mse_values):
    # Position the text slightly above the top of the highest value in the data set to avoid overlap
    y_position = max(data[i]) + 0.02  # Adjust offset as needed

    # Ensure the y_position doesn't exceed the max y-axis limit
    if y_position > 1:
        y_position = max(data[i]) * 1.02  

    ax.text(i + 1, y_position, f'MSE: {mse:.3f}', ha='center', fontsize=8)

ax.set_ylim(0.83, 1.1)  # Slightly above 1 to make space for annotations

ax.set_title('KL Divergence vs Cumulative Normalized Expected Reward')
ax.set_xlabel('KL Divergence Values')
ax.set_ylabel('Cumulative Normalized Expected Reward')
ax.set_xticklabels(x_labels)
ax.plot(range(1, len(x_labels) + 1), true[:7], linestyle='--', color='black')
plt.subplots_adjust(bottom=0.15)
plt.savefig('plot_MSE.png')
plt.show()