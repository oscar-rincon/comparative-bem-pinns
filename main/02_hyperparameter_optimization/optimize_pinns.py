#%%
# Imports y configuración
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import optuna
import joblib

# Set the current directory and utilities path
current_dir = os.getcwd()
utilities_dir = os.path.join(current_dir, '../../utilities')
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

# Importar funciones personalizadas
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement, calculate_relative_errors
from pinns_solution_functions import generate_points, MLP, init_weights, train_adam, train_lbfgs, initialize_and_load_model, predict_displacement_pinns, process_displacement_pinns
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not os.path.exists('datos'):
    os.makedirs('datos')
if not os.path.exists('models_iters'):
    os.mkdir('models_iters')

#%%
# Activación personalizada
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

#%%
# Función objetivo para Optuna

def objective(trial):
    results = []
    iter_train = 0
    adam_lr        = trial.suggest_categorical("adam_lr", [1e-2, 1e-3, 1e-4])
    hidden_layers_ = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    hidden_units_  = trial.suggest_categorical("hidden_units", [25, 50, 75])
    activation_str = trial.suggest_categorical("activation", ["Tanh", "Sigmoid", "Sine"])
    adam_fraction = 0.5
    total_iter    = 1_000
    adam_iters    = int(total_iter * adam_fraction)
    lbfgs_iters   = total_iter - adam_iters
    if activation_str == "Tanh":
        activation_function_ = nn.Tanh()
    elif activation_str == "ReLU":
        activation_function_ = nn.ReLU()
    elif activation_str == "Sigmoid":
        activation_function_ = nn.Sigmoid()
    elif activation_str == "Sine":
        activation_function_ = Sine()
    x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top = generate_points(
        n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
    )
    model = MLP(
        input_size=2,
        output_size=2,
        hidden_layers=hidden_layers_,
        hidden_units=hidden_units_,
        activation_function=activation_function_,
    ).to(device)
    model.apply(init_weights)
    start_time_adam = time.time()
    train_adam(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, adam_lr, num_iter=adam_iters
    )
    adam_training_time = time.time() - start_time_adam
    start_time_lbfgs = time.time()
    train_lbfgs(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, 1, num_iter=lbfgs_iters
    )
    lbfgs_training_time = time.time() - start_time_lbfgs
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
    return mean_rel_error_pinns

#%%
# Ejecutar Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (mean error): {best_trial.value:.3e}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

#%%
# Guardar el estudio
joblib.dump(study, "study.pkl")

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
