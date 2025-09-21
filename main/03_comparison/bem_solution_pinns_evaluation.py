
# ============================================================
"""
Script: comparison_bem_pinns.py

Description:
    This script evaluates and compares the accuracy and runtime 
    of the Boundary Element Method (BEM) and Physics-Informed 
    Neural Networks (PINNs) for solving the acoustic scattering 
    problem by a sound-hard circular obstacle.

Inputs:
    - BEM: number of boundary elements (n_values).
    - PINNs: architecture parameters (hidden layers, neurons 
      per layer), and precomputed training results stored as CSV.

Outputs (all filenames include timestamp):
    - CSV file with BEM accuracy vs. number of boundary elements,
      saved in ./data/
    - CSV file with PINNs accuracy vs. network architecture,
      saved in ./data/
    - Log file (TXT) with script runtime, saved in ./logs/
"""
# ============================================================
 
#%%
# -*- coding: utf-8 -*-
import datetime
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

# Import the function to evaluate BEM accuracy
from bem_solution_functions import evaluate_bem_accuracy
from pinns_solution_functions import set_seed, evaluate_pinn_accuracy 
set_seed(42)

#%% Start time measurement
# Record start time
start_time = time.time()

# Get current date and time string
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

#%% BEM evaluation
n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
results = []

for n in n_values:
    print(f"Evaluating for n = {n}...")
    t, err = evaluate_bem_accuracy(n=n)
    results.append({
        "n": n,
        "time_sec": t,
        "relative_error": err
    })

df = pd.DataFrame(results)

# Save with date
bem_csv = os.path.join("data", f"bem_accuracy_vs_n_{date_str}.csv")
df.to_csv(bem_csv, index=False)
print(f"Results saved to '{bem_csv}'")

# %% PINNs evaluation
layer_values = [1, 2, 3]
neuron_values = [25, 50, 75]
pinn_logs_dir = os.path.join(current_dir, "data")

results = []
for layers in layer_values:
    for neurons in neuron_values:
        print(f"Evaluating for layers = {layers}, neurons = {neurons}...")

        csv_filename = os.path.join(
            pinn_logs_dir, f"{layers}_layers_{neurons}_neurons.csv"
        )
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"Missing results file: {csv_filename}")

        metrics_df = pd.read_csv(csv_filename)
        training_time_sec = float(metrics_df["training_time_sec"].iloc[0])
        rel_error = float(metrics_df["mean_relative_error"].iloc[0])

        eval_time, _ = evaluate_pinn_accuracy(layers, neurons)

        results.append({
            "layers": layers,
            "neurons_per_layer": neurons,
            "evaluation_time_sec": eval_time,
            "relative_error": rel_error,
            "training_time_sec": training_time_sec,
        })

df = pd.DataFrame(results)

# Save with date
pinn_csv = os.path.join("data", f"pinn_accuracy_vs_architecture_{date_str}.csv")
df.to_csv(pinn_csv, index=False)
print(f"Results saved to '{pinn_csv}'")
# Save without date
pinn_csv_no_date = os.path.join("data", f"pinn_accuracy_vs_architecture.csv")
df.to_csv(pinn_csv_no_date, index=False)
print(f"Results also saved to '{pinn_csv_no_date}'")

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Get current date and time
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log filenames inside the logs folder
log_filename_with_date = os.path.join(output_folder, f"{script_name}_log_{date_str}.txt")
log_filename_no_date   = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file with date
with open(log_filename_with_date, "w") as f:
    f.write(log_text)

# Write log file without date
with open(log_filename_no_date, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename_with_date}")
print(f"Log also saved to: {log_filename_no_date}")
