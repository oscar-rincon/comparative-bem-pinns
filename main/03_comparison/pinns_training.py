# ============================================================
"""
Script: pinns_training.py

Description:
    This script performs a systematic hyperparameter study of 
    Physics-Informed Neural Networks (PINNs) applied to the 
    scattering problem by a sound-hard circular obstacle.
    For each combination of hidden layers and neurons, the 
    model is trained using Adam followed by L-BFGS, then 
    evaluated against the analytical solution.

Inputs:
    - Problem parameters: wave number k, geometry (inner radius, 
      domain size), number of training/collocation points.
    - Neural network hyperparameters: hidden layers, neurons 
      per layer, activation function, optimizer settings.
    - Analytical displacement fields for error computation.

Outputs (all filenames include timestamp):
    - Trained models (.pt) saved in ./models/
    - CSV files in ./data/ containing relative errors and 
      training times
    - Log file (TXT) with total runtime saved in ./logs/
"""
# ============================================================

#%%
import datetime
import random
import sys
import os
import time

# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the script's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch 
import torch.nn as nn
import importlib
import analytical_solution_functions
import bem_solution_functions
import pinns_solution_functions
import plotting_functions

# Reload custom modules each run
importlib.reload(analytical_solution_functions)
importlib.reload(bem_solution_functions)
importlib.reload(pinns_solution_functions)
importlib.reload(plotting_functions)

# Import custom functions
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement, calculate_relative_errors
from pinns_solution_functions import set_seed, generate_points, MLP, init_weights, train_adam, train_lbfgs, predict_displacement_pinns, process_displacement_pinns

set_seed(42)

#%% Start total time measurement
# Timestamp for outputs
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

total_start_time = time.time()

# Script name
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folders
output_folder = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(output_folder, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

#%% Parameters
r_i = np.pi / 4  # Inner radius
l_e = np.pi      # Outer radius
side_length = 2 * l_e
k = 3.0
n_Omega_P = 10_000
n_Gamma_I = 100
n_Gamma_E = 250
n_grid = 501

Y, X = np.mgrid[-l_e:l_e:n_grid*1j, -l_e:l_e:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)
u_inc_exact = mask_displacement(R_exact, r_i, l_e, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_e, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_e, u_exact)

results_all = []
iter_train = 0

adam_lr        = 1e-4
adam_fraction  = 0.5
adam_iters     = 500
lbfgs_iters    = 4500
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
activation_function_ = Sine()

x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top = generate_points(
    n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% Loop over layer and neuron values
layer_values = [1, 2, 3]
neuron_values = [25, 50, 75]

for hidden_layers_ in layer_values:
    for hidden_units_ in neuron_values:

        print(f"\n--- Training model with {hidden_layers_} layers and {hidden_units_} neurons ---")

        # Build model
        model = MLP(
            input_size=2,
            output_size=2,
            hidden_layers=hidden_layers_,
            hidden_units=hidden_units_,
            activation_function=activation_function_,
        ).to(device)
        
        model.apply(init_weights)

        # Training time measurement
        train_start_time = time.time()

        iter_train=train_adam(
            model, x_f, y_f, x_inner, y_inner, x_left, y_left,
            x_right, y_right, x_bottom, y_bottom, x_top, y_top,
            k, iter_train, results_all, adam_lr, num_iter=adam_iters
        )
        
        train_lbfgs(
            model, x_f, y_f, x_inner, y_inner, x_left, y_left,
            x_right, y_right, x_bottom, y_bottom, x_top, y_top,
            k, iter_train, results_all, lbfgs_lr=1, num_iter=lbfgs_iters
        )

        training_time = time.time() - train_start_time

        # Prediction and error calculation
        u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(
            model, l_e, r_i, k, n_grid
        )
        u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns, diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
            model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
        )
        rel_error_uscn_amp_pinns, rel_error_uscn_phase_pinns, max_diff_uscn_amp_pinns, min_diff_uscn_amp_pinns, max_diff_u_phase_pinns, min_diff_u_phase_pinns = calculate_relative_errors(
            u_scn_exact, u_exact, diff_uscn_amp_pinns,
            diff_u_scn_phase_pinns, R_exact, r_i
        )

        mean_rel_error_pinns = (rel_error_uscn_amp_pinns + rel_error_uscn_phase_pinns) / 2

        # ===== Save model =====
        # With timestamp
        model_name_ts = f"{hidden_layers_}_layers_{hidden_units_}_neurons_{date_str}.pt"
        model_path_ts = os.path.join("models", model_name_ts)
        torch.save(model.state_dict(), model_path_ts)
        print(f"Model saved to: {model_path_ts}")

        # Without timestamp
        model_name = f"{hidden_layers_}_layers_{hidden_units_}_neurons.pt"
        model_path = os.path.join("models", model_name)
        torch.save(model.state_dict(), model_path)
        print(f"Model also saved without timestamp: {model_path}")

        # ===== Save CSV results =====
        results_dict = {
            "hidden_layers": [hidden_layers_],
            "hidden_units": [hidden_units_],
            "mean_relative_error": [float(mean_rel_error_pinns)],
            "training_time_sec": [training_time],
        }

        # With timestamp
        csv_filename_ts = os.path.join(
            "data", f"{hidden_layers_}_layers_{hidden_units_}_neurons_{date_str}.csv"
        )
        pd.DataFrame(results_dict).to_csv(csv_filename_ts, index=False)
        print(f"Results saved to: {csv_filename_ts}")

        # Without timestamp
        csv_filename = os.path.join(
            "data", f"{hidden_layers_}_layers_{hidden_units_}_neurons.csv"
        )
        pd.DataFrame(results_dict).to_csv(csv_filename, index=False)
        print(f"Results also saved without timestamp: {csv_filename}")


#%% Record total runtime
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

# Include timestamp in log text too
log_text = (
    f"Script: {script_name}\n"
    f"Total execution time (s): {total_elapsed_time:.2f}\n"
)

# Define log filename with timestamp
log_filename = os.path.join(output_folder, f"{script_name}_log_{date_str}.txt")
log_filename_no_date   = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file
with open(log_filename, "w") as f:
    f.write(log_text)

# Write log file without date
with open(log_filename_no_date, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename}")
print(f"Log also saved to: {log_filename_no_date}")
 