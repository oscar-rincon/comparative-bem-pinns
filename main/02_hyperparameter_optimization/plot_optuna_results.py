 

#%%
import sys
import os
import time

# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir) 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
# Cargar el estudio
study_loaded = joblib.load("study.pkl")
print(study_loaded.best_value, study_loaded.best_params)

#%%
# DataFrame de resultados
# Después de correr el estudio
df = study_loaded.trials_dataframe(attrs=("number", "value", "params", "state"))
df.head()

#%%
# Extraer resultados de los trials
# ============================
df = study_loaded.trials_dataframe(attrs=("number", "value", "state"))
df = df[df["state"] == "COMPLETE"]
best_values = df["value"].cummin()

df_params = study_loaded.trials_dataframe(attrs=("number", "value", "params", "state"))
df_params = df_params[df_params["state"] == "COMPLETE"]
params = ["activation", "hidden_layers", "hidden_units", "adam_lr"]

# --- Load training log from CSV (iterations, loss, rel_error) ---
log_df = pd.read_csv("logs/training_log_3_layers_75_neurons.csv")  # adapt filename pattern
# Columns: iteration | loss | mean_rel_error

#%%
# Figura y gridspec
#fig = plt.figure(figsize=(7, 3.5))
fig = plt.figure(figsize=(7, 6.0))
gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.2, 0.8], hspace=0.5)

# Panel superior (historia de optimización)
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(df["number"], best_values, color="#c7c8c8ff", linewidth=1, label="Best Value", zorder=1)
ax0.scatter(df["number"], df["value"], color="#437ab0ff", s=10, label="Objective Value", zorder=2)
ax0.set_ylabel("Objective Value", fontsize=8)
ax0.set_xlabel("Trial number", fontsize=8)
ax0.set_xticks([0, 5, 10, 15, 20])
ax0.set_yticks([0.5, 0.75, 1, 1.25])
ax0.tick_params(axis="y", labelsize=7)
ax0.tick_params(axis="x", labelsize=7)

# Panel intermedio (slice plots)
gs2 = gs[1].subgridspec(1, len(params), wspace=0.4)
axes = [fig.add_subplot(gs2[0, i]) for i in range(len(params))]
norm = mcolors.Normalize(vmin=0, vmax=20)
for i, p in enumerate(params):
    sc = axes[i].scatter(
        df_params[f"params_{p}"],
        df_params["value"],
        c=df_params["number"],
        cmap="Blues",
        edgecolor="k",
        s=15,
        norm=norm,
        alpha=0.5
    )
    if p == "adam_lr":
        axes[i].set_xlabel(r"Learning rate $\alpha$", fontsize=8)
        axes[i].set_xscale("log")
    elif p == "hidden_layers":
        axes[i].set_xlabel(r"Hidden layers $L$", fontsize=8)
    elif p == "hidden_units":
        axes[i].set_xlabel(r"Neurons per layer $N$", fontsize=8)
        axes[i].set_xticks([25, 50, 75])
    elif p == "activation":
        axes[i].set_xlabel(r"Activation $\sigma$", fontsize=8)
    if i == 0:
        axes[i].set_ylabel("Objective Value", fontsize=8)
    axes[i].tick_params(axis="x", rotation=45, labelsize=7)
    axes[i].tick_params(axis="y", labelsize=7)

# Colorbar compartida
cbar = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.05, pad=0.02)
cbar.set_ticks(np.arange(0, 20 + 1, 5))
cbar.set_label("Trial number", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Panel inferior (relative error)
ax2 = fig.add_subplot(gs[2, 0])
pos = ax2.get_position()  # current position
ax2.set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height])  # move it down

transition = log_df["iteration"].max() // 2

# --- Split the curve into Adam (0 → transition) and L-BFGS (transition → end) ---
adam_mask = log_df["iteration"] <= transition
lbfgs_mask = log_df["iteration"] >= transition

# Adam segment (blue)
ax2.plot(
    log_df.loc[adam_mask, "iteration"],
    log_df.loc[adam_mask, "mean_rel_error"],
    color="#437ab0ff", linewidth=1.2, label="Adam"
)

# L-BFGS segment (gray)
ax2.plot(
    log_df.loc[lbfgs_mask, "iteration"],
    log_df.loc[lbfgs_mask, "mean_rel_error"],
    color="gray", linewidth=1.2, label="L-BFGS"
)

# Shaded areas
ax2.axvspan(0, transition, color="lightblue", alpha=0.3)
ax2.axvspan(transition, log_df["iteration"].max(), color="lightgray", alpha=0.3)

# --- Adam annotation ---
adam_idx = len(log_df) // 4
adam_x = log_df["iteration"].iloc[adam_idx]
adam_y = log_df["mean_rel_error"].iloc[adam_idx]
ax2.annotate(
    "Adam",
    xy=(adam_x, adam_y),
    xytext=(adam_x, adam_y*1.15),
    textcoords="data",
    fontsize=8,
    color="#02008dff",
    ha="center"
)

# --- LBFGS annotation ---
lbfgs_idx = 3 * len(log_df) // 4
lbfgs_x = log_df["iteration"].iloc[lbfgs_idx]
lbfgs_y = log_df["mean_rel_error"].iloc[lbfgs_idx]
ax2.annotate(
    "L-BFGS",
    xy=(lbfgs_x, lbfgs_y),
    xytext=(lbfgs_x, lbfgs_y*2.5),
    textcoords="data",
    fontsize=8,
    color="#2c2c2cff",
    ha="center",
)

ax2.set_ylabel("Relative Error", fontsize=8)
ax2.set_xlabel("Iteration", fontsize=8)
ax2.tick_params(axis="y", labelsize=7)
ax2.tick_params(axis="x", labelsize=7)

# Guardar y mostrar
plt.savefig("figures/05_hyperparameter_tunning.svg", dpi=300, bbox_inches="tight")
plt.savefig("figures/05_hyperparameter_tunning.pdf", dpi=300, bbox_inches="tight")
plt.show()

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