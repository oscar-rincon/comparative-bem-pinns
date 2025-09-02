 

#%%
# Imports
import time
import os 
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Record start time
start_time = time.time()

# Get script name
script_name = os.path.basename(__file__) 

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

#%%
# Figura y gridspec
fig = plt.figure(figsize=(7, 3.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.4)

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
pos = ax0.get_position()
ax0.set_position([pos.x0, pos.y0, 0.95 * pos.width, pos.height])

# Panel inferior (slice plots)
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

# Guardar y mostrar
plt.savefig("figures/hyperparameter_tunning.svg", dpi=300, bbox_inches="tight")
plt.savefig("figures/hyperparameter_tunning.pdf", dpi=300, bbox_inches="tight")
plt.show()

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

log_filename = os.path.splitext(script_name)[0] + "_log.txt"
with open(log_filename, "w") as f:
    f.write(log_text)

print(f"Log saved to {log_filename}")