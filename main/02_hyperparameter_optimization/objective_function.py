# %%% -------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
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
def objective(trial):
    results = []
    iter_train = 0

    # ---- Hyperparameters to optimize ----
    adam_lr = trial.suggest_categorical("adam_lr", [1e-2, 1e-3, 1e-4])
    lbfgs_lr = 1#trial.suggest_categorical("lbfgs_lr", [0.01, 0.1, 1.0])
    hidden_layers_ = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    hidden_units_  = trial.suggest_categorical("hidden_units", [25, 50, 75])
    activation_str = trial.suggest_categorical("activation", ["Tanh", "Sigmoid", "Sine"])
    adam_fraction = 0.5#trial.suggest_categorical("adam_fraction", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # Total fijo de iteraciones
    total_iter = 1_000
    adam_iters = int(total_iter * adam_fraction)
    lbfgs_iters = total_iter - adam_iters


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
    train_adam(model, x_f, y_f, x_inner, y_inner, x_left, y_left,
               x_right, y_right, x_bottom, y_bottom, x_top, y_top,
               k, iter_train, results, adam_lr, num_iter=adam_iters)
    adam_training_time = time.time() - start_time_adam

    # ---- Train with L-BFGS ----
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_f, y_f, x_inner, y_inner, x_left, y_left,
                x_right, y_right, x_bottom, y_bottom, x_top, y_top,
                k, iter_train, results, lbfgs_lr, num_iter=lbfgs_iters)
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

vis.plot_rank(study).update_layout(height=800)


# %%

vis.plot_optimization_history(study)

# %%
vis.plot_param_importances(study)

#%%



 