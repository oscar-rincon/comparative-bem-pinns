#%%
# -*- coding: utf-8 -*-
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
from analytical_solution_functions import sound_hard_circle_calc 
from analytical_solution_functions import mask_displacement
from pinns_solution_functions import evaluate_pinn_accuracy 

#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define output file path
output_file = os.path.join(output_folder, f"{script_name}_log.txt")

#%%

# List of n values to evaluate
n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Create an empty list to store the results
results = []

# Evaluate for each value of n
for n in n_values:
    print(f"Evaluating for n = {n}...")
    t, err = evaluate_bem_accuracy(n=n)
    results.append({
        "n": n,
        "time_sec": t,
        "relative_error": err
    })

# Convert the results to a DataFrame
df = pd.DataFrame(results)

# Save the results to a CSV file
df.to_csv("data/bem_accuracy_vs_n.csv", index=False)

# Final message
print("Results saved to 'bem_accuracy_vs_n.csv'")

# %%

# Define the number of layers and neurons per layer to evaluate
layer_values = [1, 2, 3]
neuron_values = [25, 50, 75]

# List of training times in seconds (must match the order of layer-neuron combinations)
training_pinn_time = [
    268.5808, 287.7863, 291.1042,  # layers = 1
    333.6271, 346.1852, 356.2399,  # layers = 2
    401.7561, 396.7530, 430.8179   # layers = 3
]

# List to store the results
results = []

# Evaluate each combination of layers and neurons
i = 0
for layers in layer_values:
    for neurons in neuron_values:
        print(f"Evaluating for layers = {layers}, neurons = {neurons}...")
        eval_time, rel_error = evaluate_pinn_accuracy(layers, neurons)
        results.append({
            "layers": layers,
            "neurons_per_layer": neurons,
            "evaluation_time_sec": eval_time,
            "relative_error": rel_error,
            "training_time_sec": training_pinn_time[i]
        })
        i += 1

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save results to CSV
df.to_csv("data/pinn_accuracy_vs_architecture.csv", index=False)

# Final message
print("Results saved to 'pinn_accuracy_vs_architecture.csv'")

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time
 
# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Define log filename inside the logs folder
log_filename = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")