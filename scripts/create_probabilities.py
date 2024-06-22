# New file that creates all datasets and runs the plot.py file.
# Computes KL divergence values and saves the per-step reward 
# values to be plotted.

def kl_divergence(p, q):
  for key in p.keys():
      if key not in q:
          q[key] = 1e-10  # Add a small value to handle missing entries in q

  kl_div = 0.0
  for k in p:
      if p[k] > 0 and q[k] > 0:
          kl_div += p[k] * np.log(p[k] / q[k])
  return kl_div

import subprocess
import numpy as np
import pickle

def run_create_dataset_script(alpha, distribution_str, script_path, seed):
    command = [
        "python3",
        script_path,
        "--save_dir=./tests/testdata",
        "--load_dir=./tests/testdata",
        "--env_name=grid",
        "--num_trajectory=400",
        "--max_trajectory_length=100",
        f"--alpha={alpha}",
        f"--distribution={distribution_str}",
        f"--seed={seed}",
        "--tabular_obs=0",
        "--force"
    ]
    subprocess.run(command)

def run_dice_estimator(alpha, dist, seed):
    command = [
        "python3",
        script_path_neural_dice,
        "--save_dir=./tests/testdata",
        "--load_dir=./tests/testdata",
        "--env_name=grid",
        "--num_trajectory=400",
        "--max_trajectory_length=100",
        f"--alpha={alpha}",  
        f"--distribution={dist}",
        f"--seed={seed}",
        "--tabular_obs=0"
    ]
    subprocess.run(command)
n = 7  # Number of distributions
script_path_create_dataset = "scripts/create_dataset.py"  # Path to create_dataset script
script_path_neural_dice = "scripts/run_neural_dice.py"  # Path to neural_dice script
script_path_plot = "scripts/plot.py"
# Generate distributions
dicts = []
# Run create_dataset script with each distribution
all_visitation_distributions = []

for i in range(0, 5):
    run_create_dataset_script(1.0, 7, script_path_create_dataset, i + 1)
with open('state_distribution.pkl', 'rb') as f:
    state_distribution = pickle.load(f)
all_visitation_distributions.append(state_distribution['state_distribution'])

for dist in range(0, n):
    visitation_per_distribution = []
    for i in range(0, 5):
        run_create_dataset_script(2/3, dist, script_path_create_dataset, i + 1)
        run_dice_estimator(2/3, dist, i + 1)
        with open('running_estimates.pkl', 'rb') as f:
            dict = pickle.load(f)
        per_step = dict['dict']
        dicts.append(per_step)
    with open('state_distribution.pkl', 'rb') as f:
        state_distribution = pickle.load(f)
    all_visitation_distributions.append(state_distribution['state_distribution'])
print(all_visitation_distributions)
kl_values = []
for distribution in range (1, len(all_visitation_distributions)):
    kl_values.append(kl_divergence(all_visitation_distributions[distribution], all_visitation_distributions[0]))
print(kl_values)
with open('kl_values.pkl', 'wb') as f:
    pickle.dump({'kl_values': kl_values}, f)
with open('per_steps_values.pkl', 'wb') as f:
    pickle.dump({'dicts': dicts}, f)
command = [
    "python3",
    script_path_plot,
]
subprocess.run(command)