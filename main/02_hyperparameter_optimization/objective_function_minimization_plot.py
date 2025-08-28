# %%% -------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch 
import torch.nn as nn
import optuna
import optuna.visualization as vis
from optuna.pruners import MedianPruner
import joblib


# %%% -------------------------------------------------------------------------
# Paths and utilities
# -----------------------------------------------------------------------------
# Current working directory
#current_dir = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Add utilities directory to Python path
sys.path.insert(0, utilities_dir)

# Import custom functions
from analytical_solution_functions import (
    sound_hard_circle_calc, 
    mask_displacement, 
    calculate_relative_errors
)
from pinns_solution_functions import (
    generate_points, 
    MLP, 
    init_weights, 
    train_adam, 
    train_lbfgs, 
    initialize_and_load_model, 
    predict_displacement_pinns, 
    process_displacement_pinns
)

# %%% -------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
r_i = np.pi / 4        # Inner radius
l_e = np.pi            # Outer radius
side_length = 2 * l_e  # Side length of the square
k = 3.0                # Wave number

n_Omega_P = 10_000     # Points inside the annular region
n_Gamma_I = 100        # Points on the inner boundary (r = r_i)
n_Gamma_E = 250        # Points on the outer boundary (r = r_e)
n_grid = 501           # Grid points in x and y

# Grid of points in the domain
Y, X = np.mgrid[-l_e:l_e:n_grid*1j, -l_e:l_e:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)  # Radial distance

# Exact displacement for a sound-hard circular obstacle
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

# Apply mask to displacements
u_inc_exact = mask_displacement(R_exact, r_i, l_e, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_e, u_scn_exact)
u_exact     = mask_displacement(R_exact, r_i, l_e, u_exact)

# %%% -------------------------------------------------------------------------
# Device and directories
# -----------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.exists('datos'):
    os.makedirs('datos')

if not os.path.exists('models_iters'):
    os.makedirs('models_iters')

# %%% -------------------------------------------------------------------------
# Custom modules
# -----------------------------------------------------------------------------
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# %%% -------------------------------------------------------------------------
# Optuna objective function
# -----------------------------------------------------------------------------
# %%% -------------------------------------------------------------------------
# Optuna objective function
# -----------------------------------------------------------------------------
def objective(trial):
    results = []
    iter_train = 0

    # ---- Hyperparameters to optimize ----
    adam_lr        = trial.suggest_categorical("adam_lr", [1e-2, 1e-3, 1e-4])
    hidden_layers_ = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    hidden_units_  = trial.suggest_categorical("hidden_units", [25, 50, 75])
    activation_str = trial.suggest_categorical("activation", ["Tanh", "Sigmoid", "Sine"])

    # ---- Iterations setup ----
    adam_fraction = 0.5
    total_iter    = 1_000
    adam_iters    = int(total_iter * adam_fraction)
    lbfgs_iters   = total_iter - adam_iters

    # ---- Activation function ----
    if activation_str == "Tanh":
        activation_function_ = nn.Tanh()
    elif activation_str == "ReLU":
        activation_function_ = nn.ReLU()
    elif activation_str == "Sigmoid":
        activation_function_ = nn.Sigmoid()
    elif activation_str == "Sine":
        activation_function_ = Sine()

    # ---- Generate training points ----
    x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, \
    x_bottom, y_bottom, x_top, y_top = generate_points(
        n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
    )

    # ---- Build and initialize model ----
    model = MLP(
        input_size=2,
        output_size=2,
        hidden_layers=hidden_layers_,
        hidden_units=hidden_units_,
        activation_function=activation_function_,
    ).to(device)
    model.apply(init_weights)

    # ---- Train with Adam ----
    start_time_adam = time.time()
    train_adam(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, adam_lr, num_iter=adam_iters
    )
    adam_training_time = time.time() - start_time_adam

    # ---- Train with L-BFGS ----
    start_time_lbfgs = time.time()
    train_lbfgs(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, 1, num_iter=lbfgs_iters
    )
    lbfgs_training_time = time.time() - start_time_lbfgs

    # ---- Evaluate model ----
    u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(
        model, l_e, r_i, k, n_grid
    )

    u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns, \
    diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
        model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
    )

    rel_error_uscn_amp_pinns, rel_error_uscn_phase_pinns, \
    max_diff_uscn_amp_pinns, min_diff_uscn_amp_pinns, \
    max_diff_u_phase_pinns, min_diff_u_phase_pinns = calculate_relative_errors(
        u_scn_exact, u_exact, diff_uscn_amp_pinns,
        diff_u_scn_phase_pinns, R_exact, r_i
    )

    # ---- Final metric ----
    mean_rel_error_pinns = (rel_error_uscn_amp_pinns + rel_error_uscn_phase_pinns) / 2

    return mean_rel_error_pinns


# %%% -------------------------------------------------------------------------
# Run Optuna study
# -----------------------------------------------------------------------------
study = optuna.create_study(directions=["minimize"],pruner=MedianPruner(n_warmup_steps=5) )
study.optimize(objective, n_trials=20)
 
 
# Save to file
joblib.dump(study, "study.pkl")
# %%%
# Later load it back
study_loaded = joblib.load("study.pkl")
print(study_loaded.best_value, study_loaded.best_params)
 
# %%%

# ============================
# Extract trial results
# ============================
df = study_loaded.trials_dataframe(attrs=("number", "value", "state"))
df = df[df["state"] == "COMPLETE"]
best_values = df["value"].cummin()

df_params = study_loaded.trials_dataframe(attrs=("number", "value", "params", "state"))
df_params = df_params[df_params["state"] == "COMPLETE"]

params = ["activation", "hidden_layers", "hidden_units", "adam_lr"]

# ============================
# Create figure and gridspec
# ============================
fig = plt.figure(figsize=(7, 3.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.4)

# ============================
# Top panel (optimization history)
# ============================
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(df["number"], best_values, color="#c7c8c8ff",
         linewidth=1, label="Best Value", zorder=1)
ax0.scatter(df["number"], df["value"], color="#437ab0ff",
            s=10, label="Objective Value", zorder=2)

ax0.set_ylabel("Objective Value", fontsize=8)
ax0.set_xlabel("Trial number", fontsize=8)

ax0.set_xticks([0, 5, 10, 15, 20])
ax0.set_yticks([0.5, 0.75, 1, 1.25])
ax0.tick_params(axis="y", labelsize=7)
ax0.tick_params(axis="x", labelsize=7)

# Adjust width manually (left, bottom, width, height in figure coordinates)
pos = ax0.get_position()
ax0.set_position([pos.x0, pos.y0, 0.95 * pos.width, pos.height])

# ============================
# Bottom panel (slice plots)
# ============================
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

    # X labels
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

    # Y label only for first plot
    if i == 0:
        axes[i].set_ylabel("Objective Value", fontsize=8)

    # Ticks
    axes[i].tick_params(axis="x", rotation=45, labelsize=7)
    axes[i].tick_params(axis="y", labelsize=7)

# ============================
# Shared colorbar
# ============================
cbar = fig.colorbar(sc, ax=axes,
                    orientation="vertical", fraction=0.05, pad=0.02)
cbar.set_ticks(np.arange(0, 20 + 1, 5))  # ticks cada 5 → incluye el 20
cbar.set_label("Trial number", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# ============================
# Save & show
# ============================
plt.savefig("figures/hyperparameter_tunning.svg", dpi=300, bbox_inches="tight")
plt.savefig("figures/hyperparameter_tunning.pdf", dpi=300, bbox_inches="tight")
plt.show()


#%%



 