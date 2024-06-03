import matplotlib.pyplot as plt
import pickle
import numpy as np

def mean_squared_error(estimates, true_values):
    return np.mean((np.array(estimates) - np.array(true_values)) ** 2)

# data_files = {
#     'edge': 'running_estimates_edge_d.pkl',
#     'uniform': 'running_estimates_uniform_d.pkl',
#     'distance': 'running_estimates_distance_d.pkl',
#     'mixed': 'running_estimates_mixed_d.pkl',
#     'close': 'running_estimates_close_d.pkl',
#     'fixed': 'running_estimates_fixed_d.pkl',
#     'far': 'running_estimates_far_d.pkl',
#     'behavior': 'running_estimates_behavior_d.pkl'
# }

# # labels = ['Random', 'Fixed', 'Mixed', 'Behavior', 'Distance', 'Edge', 'Close', 'Far']
# # values = sorted[1.3074, 4.05, 1.308, 1.64, 1.32, 1.3064, 4.02, 2.14]
# kl_values = {
#     'edge': 1.3064,
#     'uniform': 1.3074,
#     'distance': 1.32,
#     'mixed': 1.308,
#     'close': 4.02,
#     'fixed': 4.05,
#     'far': 2.14,
#     'behavior': 1.64
# }

# mse_results = {}

# for key, file in data_files.items():
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         mse = mean_squared_error(data['running_estimates'], data['true'])
#         mse_results[key] = mse

# for key, mse in mse_results.items():
#     print(f"MSE for {key}: {mse}")

# combined_tuple = {(key, kl_values[key]): mse_results[key] for key in mse_results}

# combined_tuple = dict(sorted(combined_tuple.items(), key=lambda item: item[0][1]))
# print(combined_tuple)
# with open('running_estimates_edge_d.pkl', 'rb') as f:
#     data_edge = pickle.load(f)

# steps_edge = data_edge['steps']
# running_estimates_edge = data_edge['running_estimates']
# true_edge = data_edge['true']

# with open('running_estimates_uniform_d.pkl', 'rb') as f:
#     data_unifom = pickle.load(f)

# running_estimates_uniform = data_unifom['running_estimates']
# true_uniform = data_unifom['true']

# with open('running_estimates_distance_d.pkl', 'rb') as f:
#     data_distance = pickle.load(f)

# running_estimates_distance = data_distance['running_estimates']
# true_distance = data_distance['true']

# with open('running_estimates_mixed_d.pkl', 'rb') as f:
#     data_mixed = pickle.load(f)

# running_estimates_mixed = data_mixed['running_estimates']
# true_mixed = data_mixed['true']

# with open('running_estimates_close_d.pkl', 'rb') as f:
#     data_close = pickle.load(f)

# running_estimates_close = data_close['running_estimates']
# true_close = data_close['true']

# with open('running_estimates_fixed_d.pkl', 'rb') as f:
#     data_fixed = pickle.load(f)

# running_estimates_fixed = data_fixed['running_estimates']
# true_fixed = data_fixed['true']

# with open('running_estimates_far_d.pkl', 'rb') as f:
#     data_far = pickle.load(f)

# running_estimates_far = data_far['running_estimates']
# true_far = data_far['true']

# with open('running_estimates_behavior_d.pkl', 'rb') as f:
#     data_behavior = pickle.load(f)

# running_estimates_behavior = data_behavior['running_estimates']
# true_behavior = data_behavior['true']
# # Example data
# mismatch = [key[1] for key in combined_tuple.keys()]
# values = combined_tuple.values()
# plt.figure(figsize=(10, 5))
# plt.plot(mismatch, values, label='MSE')
# plt.xlabel('KL-Divergence')
# plt.ylabel('MSE')
# plt.title('MSE over KL-Divergence')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot_MSE.png')
# plt.show()

with open('kl_values.pkl', 'rb') as f:
    kl = pickle.load(f)
kl_values = kl['kl_values']
with open('per_steps_values.pkl', 'rb') as f:
    ps = pickle.load(f)
ps_values = ps['dicts']
mse_results = {}
for i, dic in enumerate(ps_values):
    mse = mean_squared_error(dic['running_estimates'], dic['true'])
    mse_results[i] = mse

for key, mse in mse_results.items():
    print(f"MSE for {key}: {mse}")

combined_tuple = {(key, kl_values[key]): mse_results[key] for key in mse_results}

combined_tuple = dict(sorted(combined_tuple.items(), key=lambda item: item[0][1]))
print(combined_tuple)
mismatch = [key[1] for key in combined_tuple.keys()]
values = combined_tuple.values()
plt.figure(figsize=(10, 5))
plt.plot(mismatch, values, label='MSE')
plt.xlabel('KL-Divergence')
plt.ylabel('MSE')
plt.title('MSE over KL-Divergence')
plt.legend()
plt.grid(True)
plt.savefig('plot_MSE.png')
plt.show()
# plt.plot(steps_edge, true_edge, label='True Value Edge', linestyle='--', color='black')
# # labels = ['Off-Policy Uniform', 'On-Policy Uniform']
# # values = [1.3074, 1.3093]
# # background_color = '#BFBFBF'  # Replace with your desired color code
# # fig.patch.set_facecolor(background_color)
# # ax.set_facecolor(background_color)
# fig = plt.figure(facecolor='#BFBFBF')
# # Create the bar chart
# plt.bar(labels, values)
# plt.gca().set_facecolor('#BFBFBF')
# # Add titles and labels
# plt.title('State Visitation Mismatch')
# plt.xlabel('Distributions')
# plt.ylabel('Values')
# plt.savefig('results.png')
# # Display the chart
# plt.show()
# plt.figure(figsize=(10, 5))
# plt.plot(steps_edge, running_estimates_edge, label='Running Estimates Edge 1.306')
# plt.plot(steps_edge, true_edge, label='True Value Edge', linestyle='--', color='black')

# plt.plot(steps_edge, running_estimates_uniform, label='Running Estimates Uniform 1.307')
# plt.plot(steps_edge, true_uniform, label='True Value Uniform', linestyle='--', color='black')

# plt.plot(steps_edge, running_estimates_distance, label='Running Estimates Distance 1.32')
# plt.plot(steps_edge, true_distance, label='True Value Distance', linestyle='--', color='black')

# plt.plot(steps_edge, running_estimates_mixed, label='Running Estimates Mixed 1.308')
# plt.plot(steps_edge, true_mixed, label='True Value Mixed', linestyle='--', color='black')

# # plt.plot(steps_edge, running_estimates_close, label='Running Estimates Close 4.02')
# # plt.plot(steps_edge, running_estimates_fixed, label='Running Estimates Fixed 4.05')
# # plt.plot(steps_edge, running_estimates_far, label='Running Estimates Far 2.14')
# # plt.plot(steps_edge, running_estimates_behavior, label='Running Estimates Behavior 1.64')
# plt.xlabel('Steps')
# plt.ylabel('Per-Step Reward')
# plt.title('Running Estimates vs True Value Over Steps')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot_Small_Mismatch.png')
# plt.show()

# plt.figure(figsize=(10, 5))
# # plt.plot(steps_edge, running_estimates_edge, label='Running Estimates Edge 1.306')
# # plt.plot(steps_edge, running_estimates_uniform, label='Running Estimates Uniform 1.307')
# # plt.plot(steps_edge, running_estimates_distance, label='Running Estimates Distance 1.32')
# # plt.plot(steps_edge, running_estimates_mixed, label='Running Estimates Mixed 1.308')
# # plt.plot(steps_edge, running_estimates_close, label='Running Estimates Close 4.02')
# # plt.plot(steps_edge, running_estimates_fixed, label='Running Estimates Fixed 4.05')
# plt.plot(steps_edge, running_estimates_far, label='Running Estimates Far 2.14')
# plt.plot(steps_edge, true_far, label='True Value Far', linestyle='--', color='black')

# plt.plot(steps_edge, running_estimates_behavior, label='Running Estimates Behavior 1.64')
# plt.plot(steps_edge, true_behavior, label='True Value Behavior', linestyle='--', color='black')
# plt.xlabel('Steps')
# plt.ylabel('Per-Step Reward')
# plt.title('Running Estimates vs True Value Over Steps')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot_Mid_Mismatch.png')
# plt.show()

# plt.figure(figsize=(10, 5))
# # plt.plot(steps_edge, running_estimates_edge, label='Running Estimates Edge 1.306')
# # plt.plot(steps_edge, running_estimates_uniform, label='Running Estimates Uniform 1.307')
# # plt.plot(steps_edge, running_estimates_distance, label='Running Estimates Distance 1.32')
# # plt.plot(steps_edge, running_estimates_mixed, label='Running Estimates Mixed 1.308')
# plt.plot(steps_edge, running_estimates_close, label='Running Estimates Close 4.02')
# plt.plot(steps_edge, true_close, label='True Value Close', linestyle='--', color='black')

# plt.plot(steps_edge, running_estimates_fixed, label='Running Estimates Fixed 4.05')
# plt.plot(steps_edge, true_fixed, label='True Value Fixed', linestyle='--', color='red')

# # plt.plot(steps_edge, running_estimates_far, label='Running Estimates Far 2.14')
# # plt.plot(steps_edge, running_estimates_behavior, label='Running Estimates Behavior 1.64')
# # plt.plot(steps_edge, true_edge, label='True Value', linestyle='--', color='black')
# plt.xlabel('Steps')
# plt.ylabel('Per-Step Reward')
# plt.title('Running Estimates vs True Value Over Steps')
# plt.legend()
# plt.grid(True)
# plt.savefig('plot_High_Mismatch.png')
# plt.show()

# # plt.figure(figsize=(10, 5))
# # plt.plot(steps_edge, running_estimates_edge, label='Running Estimates Edge 1.306')
# # plt.plot(steps_edge, running_estimates_uniform, label='Running Estimates Uniform 1.307')
# # plt.plot(steps_edge, running_estimates_distance, label='Running Estimates Distance 1.32')
# # plt.plot(steps_edge, running_estimates_mixed, label='Running Estimates Mixed 1.308')
# # plt.plot(steps_edge, running_estimates_close, label='Running Estimates Close 4.02')
# # plt.plot(steps_edge, running_estimates_fixed, label='Running Estimates Fixed 4.05')
# # plt.plot(steps_edge, running_estimates_far, label='Running Estimates Far 2.14')
# # plt.plot(steps_edge, running_estimates_behavior, label='Running Estimates Behavior 1.64')
# # plt.plot(steps_edge, true_edge, label='True Value', linestyle='--', color='black')
# # plt.xlabel('Steps')
# # plt.ylabel('Per-Step Reward')
# # plt.title('Running Estimates vs True Value Over Steps')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('plot_Every_Mismatch.png')
# # plt.show()
