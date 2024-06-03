# import numpy as np

# def generate_random_distribution(k, seed=None):
#     if seed is not None:
#         np.random.seed(seed)  # Set the seed for reproducibility
#     # Generate k random positive numbers
#     random_numbers = np.random.rand(k)
    
#     # Normalize the numbers to make their sum equal to 1
#     probability_distribution = random_numbers / np.sum(random_numbers)
    
#     return probability_distribution

# def generate_multiple_distributions(k, n, global_seed):
#     np.random.seed(global_seed)  # Set the global seed for reproducibility
#     seeds = np.random.randint(0, 1e9, size=n)  # Generate n unique seeds
    
#     distributions = []
#     for seed in seeds:
#         distribution = generate_random_distribution(k, seed)
#         distributions.append(distribution)
    
#     return distributions

# # Example usage
# k = 16
# n = 100
# global_seed = 42
# distributions = generate_multiple_distributions(k, n, global_seed)

# # Print a few distributions
# for i, dist in enumerate(distributions):
#     print(f"Distribution {i+1}: {dist}")
#     print(f"Sum of probabilities: {np.sum(dist)}")

# # If you run this again with the same global_seed, it will produce the same distributions
# distributions_reproduced = generate_multiple_distributions(k, n, global_seed)

# # Verify that the distributions are the same
# for i in range(n):
#     assert np.allclose(distributions[i], distributions_reproduced[i]), f"Mismatch in distribution {i+1}"

# print("All distributions are reproducible and match!")
import subprocess
import numpy as np
import pickle
# def generate_random_distribution(k, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     random_numbers = np.random.rand(k)
#     probability_distribution = random_numbers / np.sum(random_numbers)
#     return probability_distribution

# def generate_multiple_distributions(k, n, global_seed):
#     np.random.seed(global_seed)  # Set the global seed for reproducibility
#     seeds = np.random.randint(0, 1e9, size=n)  # Generate n unique seeds
    
#     distributions = []
#     for i, seed in enumerate(seeds):
#         distribution = generate_random_distribution(k, seed)
#         distributions.append(distribution)
    
#     return distributions

# def run_create_dataset_script(distribution_str, script_path):
def run_create_dataset_script(script_path):
    alpha_values = [0.0, 1.0]
    # distribution_str = ",".join(map(str, distribution_value))
    for alpha in alpha_values:
        command = [
            "python3",
            script_path,
            "--save_dir=./tests/testdata",
            # "--load_dir=./tests/testdata/Grid",
            "--load_dir=./tests/testdata",
            "--env_name=grid",
            "--num_trajectory=40",
            "--max_trajectory_length=10",
            f"--alpha={alpha}",
            # f"--distribution={distribution_str}",
            "--tabular_obs=0",
            "--force"
        ]
        subprocess.run(command)

# Example usage
k = 100  # Number of points in each distribution
n = 1  # Number of distributions
# global_seed = 42  # Global seed for reproducibility
script_path_create_dataset = "scripts/create_dataset.py"  # Path to create_dataset script
script_path_neural_dice = "scripts/run_neural_dice.py"  # Path to neural_dice script

# Generate distributions
# distributions = generate_multiple_distributions(k, n, global_seed)
kl_values = []
dicts = []
# Run create_dataset script with each distribution
# for dist in distributions:
for dist in range(0, n):
    # distribution_str = ",".join(map(str, dist))
    # run_create_dataset_script(distribution_str, script_path_create_dataset)
    run_create_dataset_script(script_path_create_dataset)
    with open('distibution_test_d.pkl', 'rb') as f:
        mismatch = pickle.load(f)
    kl = mismatch['mismatch']
    kl_values.append(kl)
    command = [
        "python3",
        script_path_neural_dice,
        "--save_dir=./tests/testdata",
        "--load_dir=./tests/testdata",
        "--env_name=grid",
        "--num_trajectory=40",
        "--max_trajectory_length=10",
        "--alpha=0.0",  # Assuming alpha is always 0.0 for neural_dice
        # f"--distribution={distribution_str}",
        "--tabular_obs=0"
    ]
    subprocess.run(command)
    with open('running_estimates_test_d.pkl', 'rb') as f:
        dict = pickle.load(f)
    per_step = dict['dict']
    dicts.append(per_step)

with open('kl_values.pkl', 'wb') as f:
    pickle.dump({'kl_values': kl_values}, f)
with open('per_steps_values.pkl', 'wb') as f:
    pickle.dump({'dicts': dicts}, f)