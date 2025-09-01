 
#%%
# -*- coding: utf-8 -*-
import sys
import os
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


#%%

# # List of n values to evaluate
# n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# # Create an empty list to store the results
# results = []

# # Evaluate for each value of n
# for n in n_values:
#     print(f"Evaluating for n = {n}...")
#     t, err = evaluate_bem_accuracy(n=n)
#     results.append({
#         "n": n,
#         "time_sec": t,
#         "relative_error": err
#     })

# # Convert the results to a DataFrame
# df = pd.DataFrame(results)

# # Save the results to a CSV file
# df.to_csv("data/bem_accuracy_vs_n.csv", index=False)

# # Final message
# print("Results saved to 'bem_accuracy_vs_n.csv'")
#%%
 
# # Define the number of layers and neurons per layer to evaluate
# layer_values = [1, 2, 3]
# neuron_values = [25, 50, 75]

# # List of training times in seconds (must match the order of layer-neuron combinations)
# training_pinn_time = [
#     268.5808, 287.7863, 291.1042,  # layers = 1
#     333.6271, 346.1852, 356.2399,  # layers = 2
#     401.7561, 396.7530, 430.8179   # layers = 3
# ]

# # List to store the results
# results = []

# # Evaluate each combination of layers and neurons
# i = 0
# for layers in layer_values:
#     for neurons in neuron_values:
#         print(f"Evaluating for layers = {layers}, neurons = {neurons}...")
#         eval_time, rel_error = evaluate_pinn_accuracy(layers, neurons)
#         results.append({
#             "layers": layers,
#             "neurons_per_layer": neurons,
#             "evaluation_time_sec": eval_time,
#             "relative_error": rel_error,
#             "training_time_sec": training_pinn_time[i]
#         })
#         i += 1

# # Convert results to a DataFrame
# df = pd.DataFrame(results)

# # Save results to CSV
# df.to_csv("data/pinn_accuracy_vs_architecture.csv", index=False)

# # Final message
# print("Results saved to 'pinn_accuracy_vs_architecture.csv'")

#%%

# --- Load CSV data ---
bem_df = pd.read_csv("data/bem_accuracy_vs_n.csv")
pinn_df = pd.read_csv("data/pinn_accuracy_vs_architecture.csv")
 
#%%

# Values reported in the paragraph
# PINN with 3 layers and 75 neurons
pinn_val = pinn_df[(pinn_df["layers"] == 3) & (pinn_df["neurons_per_layer"] == 75)].iloc[0]

# Closest BEM (n=15)
bem_val = bem_df[bem_df["n"] == 15].iloc[0]

# Relative speed calculations
bem_vs_pinn_training = bem_val["time_sec"] / pinn_val["training_time_sec"]  # ratio < 1
pinn_eval_vs_bem = bem_val["time_sec"] / pinn_val["evaluation_time_sec"]    # ratio > 1

# Express as "times faster"
bem_times_faster = round(1 / bem_vs_pinn_training)   
pinn_times_faster = round(pinn_eval_vs_bem)         

# Collect reported values
reported_values = {
    "PINN (3,75) relative_error": pinn_val["relative_error"],
    "PINN (3,75) training_time_sec": pinn_val["training_time_sec"],
    "PINN (3,75) evaluation_time_sec": pinn_val["evaluation_time_sec"],
    "BEM (n=15) relative_error": bem_val["relative_error"],
    "BEM (n=15) time_sec": bem_val["time_sec"],
    "BEM ~times faster than PINN training": bem_times_faster,
    "PINN evaluation ~times faster than BEM": pinn_times_faster,
}

reported_values


 
#%%

# --- Marker sizes ---
pinn_marker_sizes = 7 #* pinn_df["layers"]  # PINN: scaled by number of layers
bem_marker_sizes = 7 #* bem_df["n"]          # BEM: scaled by number of integration points

# --- Plot setup ---
plt.figure(figsize=(6.7, 2.0))

# --- Plot BEM (blue) ---
bem_points = plt.scatter(bem_df["relative_error"], bem_df["time_sec"],
                         color="#437ab0ff", edgecolors="#437ab0ff",
                         label='BEM (solution)', s=bem_marker_sizes, zorder=5)
 

# --- Plot PINN evaluation time (gray) ---
pinn_eval_points = plt.scatter(pinn_df["relative_error"], pinn_df["evaluation_time_sec"],
                                color="#5e5e5e", edgecolors="#5e5e5e",
                                label='PINN (evaluation)', s=pinn_marker_sizes, zorder=4)

 
# --- Plot PINN training time (black) ---
pinn_train_points = plt.scatter(pinn_df["relative_error"], pinn_df["training_time_sec"],
                                 color="#000000", edgecolors="#000000",
                                 label='PINN (training)', s=pinn_marker_sizes, zorder=3)
 
# --- Axes labels ---
plt.xlabel('Relative Error', fontsize=8)
plt.ylabel('Time (s)', fontsize=8)

# --- Log-log scale ---
plt.xscale('log')
plt.yscale('log')

# --- Ticks ---
ax = plt.gca()
#ax.grid(True, which="major", axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.6)
ax.set_xticks([1e+0, 1e-1, 1e-2])
ax.set_xticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
ax.set_yticks([1e+3, 1e+2, 1e+1, 1e+0, 1e-1, 1e-2])
ax.set_yticklabels([r'$10^{3}$', r'$10^{2}$', r'$10^{1}$', r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)

# --- Grid ---
#plt.grid(True, which="both", ls="--", linewidth=0.5)

# --- Legend ---
plt.legend(loc='lower left', fontsize=7.5, frameon=False,
           handletextpad=0.5, markerscale=0.9, labelspacing=1.2)

# --- Final layout and save ---
plt.tight_layout()
plt.savefig("figures/rel_error_time.svg", dpi=150, bbox_inches='tight')
plt.show()
 