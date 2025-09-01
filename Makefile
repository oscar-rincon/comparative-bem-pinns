# Makefile to run Python scripts in sequence

.PHONY: all clean run_all \
	run_analytical_solution \
	run_hyperparameter_optimization \
	run_plot_optuna_results \
	run_comparison_bem run_comparison_pinns \
	run_comparison_plot_a run_comparison_plot_b \
	run_generalization_bem run_generalization_pinns \
	run_generalization_plot

# Default target
all: run_all

# Run all scripts in order
run_all: run_analytical_solution run_hyperparameter_optimization \
         run_plot_optuna_results run_comparison_bem run_comparison_pinns \
         run_comparison_plot_a run_comparison_plot_b \
         run_generalization_bem run_generalization_pinns run_generalization_plot

# Analytical solution
run_analytical_solution:
	@echo "Running analytical_solution.py..."
	@python main/01_analytical_solution/analytical_solution.py
	@echo "Finished analytical_solution.py..."

# Hyperparameter optimization
run_hyperparameter_optimization:
	@echo "Running optimize_pinns.py..."
	@python main/02_hyperparameter_optimization/optimize_pinns.py
	@echo "Finished optimize_pinns.py..."

# Plot Optuna results
run_plot_optuna_results:
	@echo "Running plot_optuna_results.py..."
	@python main/02_hyperparameter_optimization/plot_optuna_results.py
	@echo "Finished plot_optuna_results.py..."

# Comparison BEM
run_comparison_bem:
	@echo "Running comparison_bem.py..."
	@python main/03_comparison/comparison_bem.py
	@echo "Finished comparison_bem.py..."

# Comparison PINNs
run_comparison_pinns:
	@echo "Running comparison_pinns.py..."
	@python main/03_comparison/comparison_pinns.py
	@echo "Finished comparison_pinns.py..."

# Comparison plots A
run_comparison_plot_a:
	@echo "Running comparison_plot_a.py..."
	@python main/03_comparison/comparison_plot_a.py
	@echo "Finished comparison_plot_a.py..."

# Comparison plots B
run_comparison_plot_b:
	@echo "Running comparison_plot_b.py..."
	@python main/03_comparison/comparison_plot_b.py
	@echo "Finished comparison_plot_b.py..."

# Generalization BEM
run_generalization_bem:
	@echo "Running generalization_bem.py..."
	@python main/04_generalization/generalization_bem.py
	@echo "Finished generalization_bem.py..."

# Generalization PINNs
run_generalization_pinns:
	@echo "Running generalization_pinns.py..."
	@python main/04_generalization/generalization_pinns.py
	@echo "Finished generalization_pinns.py..."

# Generalization plot
run_generalization_plot:
	@echo "Running generalization_plot.py..."
	@python main/04_generalization/generalization_plot.py
	@echo "Finished generalization_plot.py..."

# Clean up figures
clean:
	@echo "Cleaning up figures..."
	@rm -rf main/01_analytical_solution/figures/*
	@rm -rf main/02_hyperparameter_optimization/figures/*
	@rm -rf main/03_comparison/figures/*
	@rm -rf main/04_generalization/figures/*
	@echo "Clean up complete."
