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
import torch 
import torch.nn as nn
import importlib
import analytical_solution_functions
import bem_solution_functions
import pinns_solution_functions
import plotting_functions

# Reload them each time this file runs
importlib.reload(analytical_solution_functions)
importlib.reload(bem_solution_functions)
importlib.reload(pinns_solution_functions)
importlib.reload(plotting_functions)

# Importar funciones personalizadas
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement, calculate_relative_errors
from pinns_solution_functions import generate_points, MLP, init_weights, train_adam_with_logs, train_lbfgs_with_logs, initialize_and_load_model, predict_displacement_pinns, process_displacement_pinns


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

# Parámetros
r_i = np.pi / 4  # Radio interno
l_e = np.pi      # Radio externo
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

results = []
iter_train = 0

adam_lr        = 1e-3
hidden_layers_ = 3
hidden_units_  = 75
adam_fraction  = 0.5
adam_iters     = 5000
lbfgs_iters    = 5000
activation_function_ = nn.Tanh()

x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top = generate_points(
    n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MLP(
    input_size=2,
    output_size=2,
    hidden_layers=hidden_layers_,
    hidden_units=hidden_units_,
    activation_function=activation_function_,
).to(device)

model.apply(init_weights)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Build CSV filename
csv_name = f"logs/training_log_{hidden_layers_}_layers_{hidden_units_}_neurons.csv"

# --- Adam training with logs ---
iter_train = train_adam_with_logs(
    model,
    x_f, y_f,
    x_inner, y_inner,
    x_left, y_left,
    x_right, y_right,
    x_bottom, y_bottom,
    x_top, y_top,
    k,
    iter_train,
    results,
    adam_lr,
    num_iter=adam_iters,
    save_csv_path=csv_name,  # dynamic filename
    l_e=l_e,
    r_i=r_i,
    n_grid=n_grid,
    X=X,
    Y=Y,
    R_exact=R_exact,
    u_scn_exact=u_scn_exact,
    u_exact=u_exact
)

# --- L-BFGS training with logs (continues iteration count) ---
iter_train = train_lbfgs_with_logs(
    model,
    x_f, y_f,
    x_inner, y_inner,
    x_left, y_left,
    x_right, y_right,
    x_bottom, y_bottom,
    x_top, y_top,
    k,
    iter_start=iter_train,   # continue from Adam
    results=results,
    lbfgs_lr=1.0,
    num_iter=lbfgs_iters,
    save_csv_path=csv_name,  # same CSV, continues appending
    l_e=l_e,
    r_i=r_i,
    n_grid=n_grid,
    X=X,
    Y=Y,
    R_exact=R_exact,
    u_scn_exact=u_scn_exact,
    u_exact=u_exact
)

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