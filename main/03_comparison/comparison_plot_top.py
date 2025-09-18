 
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

# --- Load CSV data ---
bem_df = pd.read_csv("data/bem_accuracy_vs_n.csv")
pinn_df = pd.read_csv("data/pinn_accuracy_vs_architecture.csv")
 
#%%

# Values reported in the paragraph
# PINN with 3 layers and 75 neurons
pinn_val = pinn_df[(pinn_df["layers"] == 3) & (pinn_df["neurons_per_layer"] == 75)].iloc[0]

# Closest BEM (n=15)
bem_val = bem_df[bem_df["n"] == 15].iloc[0]

# --- Extract selected values for plotting ---
error_bem_sel = bem_val["relative_error"]
time_bem_sel  = bem_val["time_sec"]

error_pinn_sel = pinn_val["relative_error"]
time_pinn_sel  = pinn_val["training_time_sec"]  
time_pinn_eval = pinn_val["evaluation_time_sec"]

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

# Save dictionary to txt
with open("logs/reported_values.txt", "w") as f:
    f.write("Reported values\n")
    f.write("=====================\n")
    for key, val in reported_values.items():
        f.write(f"{key}: {val}\n")

 
#%%

# --- Marker sizes ---
pinn_marker_sizes = 7 #* pinn_df["layers"]  # PINN: scaled by number of layers
bem_marker_sizes = 7 #* bem_df["n"]          # BEM: scaled by number of integration points

# --- Plot setup ---
plt.figure(figsize=(6.7, 2.0))

# # --- Plot BEM (blue) ---
# bem_points = plt.scatter(bem_df["relative_error"], bem_df["time_sec"],
#                          color="#437ab0ff", edgecolors="#437ab0ff",
#                          label='BEM (solution)', s=bem_marker_sizes, zorder=3)
 

# # --- Plot PINN evaluation time (gray) ---
# pinn_eval_points = plt.scatter(pinn_df["relative_error"], pinn_df["evaluation_time_sec"],
#                                 color="#5e5e5e", edgecolors="#5e5e5e",
#                                 label='PINN (evaluation)', s=pinn_marker_sizes, zorder=3)

 
# # --- Plot PINN training time (black) ---
# pinn_train_points = plt.scatter(pinn_df["relative_error"], pinn_df["training_time_sec"],
#                                  color="#000000", edgecolors="#000000",
#                                  label='PINN (training)', s=pinn_marker_sizes, zorder=3)

# --- Marker sizes ---
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


# ==========================
# HIGHLIGHTS (same marker, hollow)
# ==========================

# PINN highlights
manual_highlights = [
    #{"L": 1, "n": 75, "color": "#00a2ff"},
    #{"L": 2, "n": 50, "color": "#ee2d2d"},
    {"L": 3, "n": 75, "color": "#00ff0d"},
]

for h in manual_highlights:
    row = pinn_df[(pinn_df["layers"] == h["L"]) &
                  (pinn_df["neurons_per_layer"] == h["n"])].iloc[0]
    # training
    plt.scatter(row["relative_error"], row["training_time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=pinn_marker_sizes, linewidths=1.0, zorder=4)
    # evaluation
    plt.scatter(row["relative_error"], row["evaluation_time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=pinn_marker_sizes, linewidths=1.0, zorder=4)

# BEM highlights
bem_highlights = [
    #{"n": 5, "color": "#00a2ff"},
    #{"n": 10, "color": "#ee2d2d"},
    {"n": 15, "color": "#00ff0d"},
]

for h in bem_highlights:
    row = bem_df[bem_df["n"] == h["n"]].iloc[0]
    plt.scatter(row["relative_error"], row["time_sec"],
                facecolors="none", edgecolors=h["color"],
                s=bem_marker_sizes, linewidths=0.6, zorder=4)

# # --- Horizontal lines for BEM ---
bem_min = bem_df.loc[bem_df["time_sec"].idxmin()]
bem_max = bem_df.loc[bem_df["time_sec"].idxmax()]

# # --- Horizontal lines for PINN evaluation ---
pinn_eval_min = pinn_df.loc[pinn_df["evaluation_time_sec"].idxmin()]
pinn_eval_max = pinn_df.loc[pinn_df["evaluation_time_sec"].idxmax()]

# # --- Horizontal lines for PINN training ---
pinn_train_min = pinn_df.loc[pinn_df["training_time_sec"].idxmin()]
pinn_train_max = pinn_df.loc[pinn_df["training_time_sec"].idxmax()]

# --- Gray transparent backgrounds instead of horizontal lines ---
# BEM range
plt.axhspan(bem_min["time_sec"], bem_max["time_sec"],
            color="#437ab0", alpha=0.1, zorder=0)

# PINN evaluation range
plt.axhspan(pinn_eval_min["evaluation_time_sec"], pinn_eval_max["evaluation_time_sec"],
            color="#5e5e5e", alpha=0.1, zorder=0)

# PINN training range
plt.axhspan(pinn_train_min["training_time_sec"], pinn_train_max["training_time_sec"],
            color="#000000", alpha=0.07, zorder=0)

# --- Print values instead of plotting them ---
print("\n--- Min/Max Times ---")
print(f"BEM:   min={bem_min['time_sec']:.3f} s (n={int(bem_min['n'])}), "
      f"max={bem_max['time_sec']:.3f} s (n={int(bem_max['n'])})")

print(f"PINN Evaluation: min={pinn_eval_min['evaluation_time_sec']:.3f} s "
      f"(L={pinn_eval_min['layers']}, n={pinn_eval_min['neurons_per_layer']}), "
      f"max={pinn_eval_max['evaluation_time_sec']:.3f} s "
      f"(L={pinn_eval_max['layers']}, n={pinn_eval_max['neurons_per_layer']})")

print(f"PINN Training:   min={pinn_train_min['training_time_sec']:.3f} s "
      f"(L={pinn_train_min['layers']}, n={pinn_train_min['neurons_per_layer']}), "
      f"max={pinn_train_max['training_time_sec']:.3f} s "
      f"(L={pinn_train_max['layers']}, n={pinn_train_max['neurons_per_layer']})")

# --- Axes labels ---
plt.xlabel('Relative Error', fontsize=8)
plt.ylabel('Time (s)', fontsize=8)

# --- Log-log scale ---
plt.xscale('log')
plt.yscale('log')


# --- Legend ---
#plt.legend(loc='lower left', fontsize=7.5, frameon=False,
#           handletextpad=0.1, markerscale=1.0, labelspacing=0.8)



# --- Ticks ---
ax = plt.gca()

# Use the right edge of the plot for placement
x_text = ax.get_xlim()[1] * 1.05  # little outside the right edge

# Labels aligned with ranges
ax.text(x_text, (bem_min["time_sec"] * bem_max["time_sec"])**0.5,
        "BEM (solution)", va="center", ha="left",
        fontsize=7.5, color="#437ab0ff")

ax.text(x_text, (pinn_eval_min["evaluation_time_sec"] * pinn_eval_max["evaluation_time_sec"])**0.5,
        "PINN (evaluation)", va="center", ha="left",
        fontsize=7.5, color="#5e5e5e")

ax.text(x_text, (pinn_train_min["training_time_sec"] * pinn_train_max["training_time_sec"])**0.5,
        "PINN (training)", va="center", ha="left",
        fontsize=7.5, color="#000000")

#ax.grid(True, which="major", axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.6)
ax.set_xticks([1e+0, 1e-1, 1e-2])
ax.set_xticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=10))
ax.set_yticks([1e+3, 1e+2, 1e+1, 1e+0, 1e-1, 1e-2])
ax.set_yticklabels([r'$10^{3}$', r'$10^{2}$', r'$10^{1}$', r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$'], fontsize=8)

# --- Final layout and save ---
plt.tight_layout()
plt.savefig("figures/rel_error_time.svg", dpi=150, bbox_inches='tight')
plt.savefig("figures/rel_error_time.pdf", dpi=150, bbox_inches='tight')
#plt.show()
 
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
# %%
