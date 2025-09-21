# ============================================================
"""
Script: comparison_plot_top.py

Description:
    This script compares the computational cost and accuracy of 
    the Boundary Element Method (BEM) and Physics-Informed Neural 
    Networks (PINNs). It loads benchmark results from CSV files, 
    highlights representative cases (BEM with n=15, PINN with L=3, 
    N=75), and generates log-log scatter plots of relative error vs. 
    runtime.

Inputs:
    - data/bem_accuracy_vs_n.csv
    - data/pinn_accuracy_vs_architecture.csv

Outputs:
    - Figures: 
        figures/rel_error_time.svg
        figures/rel_error_time.pdf
    - reported_values_<timestamp>.txt (summary of selected values, in ./data/)
    - Log file (TXT) with script name, execution time, and timestamps, saved in ./logs/
"""
# ============================================================

#%%
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

# Change the working directory to the script directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)


#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(output_folder, exist_ok=True)

# Define figures folder
figures_folder = os.path.join(current_dir, "figures")
os.makedirs(figures_folder, exist_ok=True)

# Define data folder
data_folder = os.path.join(current_dir, "data")
os.makedirs(data_folder, exist_ok=True)


#%% Load CSV data
bem_df = pd.read_csv("data/bem_accuracy_vs_n.csv")
pinn_df = pd.read_csv("data/pinn_accuracy_vs_architecture.csv")


#%% Extract reported values
# PINN with 3 layers and 75 neurons
pinn_val = pinn_df[(pinn_df["layers"] == 3) & (pinn_df["neurons_per_layer"] == 75)].iloc[0]

# Closest BEM (n=15)
bem_val = bem_df[bem_df["n"] == 15].iloc[0]

# Extract values
error_bem_sel = bem_val["relative_error"]
time_bem_sel  = bem_val["time_sec"]

error_pinn_sel = pinn_val["relative_error"]
time_pinn_sel  = pinn_val["training_time_sec"]
time_pinn_eval = pinn_val["evaluation_time_sec"]

# Relative speed calculations
bem_vs_pinn_training = bem_val["time_sec"] / pinn_val["training_time_sec"]
pinn_eval_vs_bem = bem_val["time_sec"] / pinn_val["evaluation_time_sec"]

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

# Save reported values with timestamp
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
reported_file = os.path.join("data", f"reported_values_{date_str}.txt")

with open(reported_file, "w") as f:
    f.write(f"Reported values generated on {date_str}\n")
    f.write("====================================\n")
    for key, val in reported_values.items():
        f.write(f"{key}: {val}\n")

print(f"Reported values saved to: {reported_file}")


#%% Plotting
pinn_marker_sizes = 15
bem_marker_sizes = 15

plt.figure(figsize=(6.7, 2.0))

# --- Plot BEM (blue, all points) ---
plt.scatter(bem_df["relative_error"], bem_df["time_sec"],
            facecolors="#437ab0ff", edgecolors="#437ab0ff",
            label='BEM (solution)', s=bem_marker_sizes, zorder=3)

# --- Plot PINN evaluation (gray, all points) ---
plt.scatter(pinn_df["relative_error"], pinn_df["evaluation_time_sec"],
            facecolors="#5e5e5e", edgecolors="#5e5e5e",
            label='PINN (evaluation)', s=pinn_marker_sizes, zorder=3)

# --- Plot PINN training (black, all points) ---
plt.scatter(pinn_df["relative_error"], pinn_df["training_time_sec"],
            facecolors="#000000", edgecolors="#000000",
            label='PINN (training)', s=pinn_marker_sizes, zorder=3)

# --- Highlights ---
manual_highlights = [{"L": 3, "n": 75, "color": "#00ff0d"}]
for h in manual_highlights:
    row = pinn_df[(pinn_df["layers"] == h["L"]) &
                  (pinn_df["neurons_per_layer"] == h["n"])].iloc[0]
    plt.scatter(row["relative_error"], row["training_time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=pinn_marker_sizes, linewidths=1.0, zorder=4)
    plt.scatter(row["relative_error"], row["evaluation_time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=pinn_marker_sizes, linewidths=1.0, zorder=4)

bem_highlights = [{"n": 15, "color": "#00ff0d"}]
for h in bem_highlights:
    row = bem_df[bem_df["n"] == h["n"]].iloc[0]
    plt.scatter(row["relative_error"], row["time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=bem_marker_sizes, linewidths=0.6, zorder=4)

# --- Shaded ranges ---
bem_min = bem_df.loc[bem_df["time_sec"].idxmin()]
bem_max = bem_df.loc[bem_df["time_sec"].idxmax()]
pinn_eval_min = pinn_df.loc[pinn_df["evaluation_time_sec"].idxmin()]
pinn_eval_max = pinn_df.loc[pinn_df["evaluation_time_sec"].idxmax()]
pinn_train_min = pinn_df.loc[pinn_df["training_time_sec"].idxmin()]
pinn_train_max = pinn_df.loc[pinn_df["training_time_sec"].idxmax()]

plt.axhspan(bem_min["time_sec"], bem_max["time_sec"], color="#437ab0", alpha=0.1, zorder=0)
plt.axhspan(pinn_eval_min["evaluation_time_sec"], pinn_eval_max["evaluation_time_sec"],
            color="#5e5e5e", alpha=0.1, zorder=0)
plt.axhspan(pinn_train_min["training_time_sec"], pinn_train_max["training_time_sec"],
            color="#000000", alpha=0.07, zorder=0)

# --- Labels and axes ---
plt.xlabel('Relative Error', fontsize=8)
plt.ylabel('Time (s)', fontsize=8)
plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
x_text = ax.get_xlim()[1] * 1.05
ax.text(x_text, (bem_min["time_sec"] * bem_max["time_sec"])**0.5,
        "BEM (solution)", va="center", ha="left", fontsize=7.5, color="#437ab0ff")
ax.text(x_text, (pinn_eval_min["evaluation_time_sec"] * pinn_eval_max["evaluation_time_sec"])**0.5,
        "PINN (evaluation)", va="center", ha="left", fontsize=7.5, color="#5e5e5e")
ax.text(x_text, (pinn_train_min["training_time_sec"] * pinn_train_max["training_time_sec"])**0.5,
        "PINN (training)", va="center", ha="left", fontsize=7.5, color="#000000")

ax.set_xticks([1e+0, 1e-1, 1e-2])
ax.set_xticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
ax.set_yticks([1e+3, 1e+2, 1e+1, 1e+0, 1e-1, 1e-2])
ax.set_yticklabels([r'$10^{3}$', r'$10^{2}$', r'$10^{1}$', r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "rel_error_time.svg"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(figures_folder, "rel_error_time.pdf"), dpi=150, bbox_inches='tight')
# plt.show()


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