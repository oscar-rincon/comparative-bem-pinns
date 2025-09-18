# Makefile to run Optuna minimization script and train optimized PINNs

.PHONY: all optimize train plot_optimize clean

all: optimize train plot_optimize

# Run hyperparameter optimization
optimize:
	@echo "Running optimize_pinns.py with Optuna..."
	@python3 optimize_pinns.py
	@echo "Finished optimize_pinns.py."

# Train with optimized hyperparameters
train:
	@echo "Running trainning_optimized_pinns.py..."
	@python3 trainning_optimized_pinns.py
	@echo "Finished trainning_optimized_pinns.py."

# Run plotting script
plot_optimize:
	@echo "Running plot_optuna_results.py..."
	@python3 plot_optuna_results.py
	@echo "Finished plot_optuna_results.py."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* || true
	@rm -rf __pycache__ || true
	@rm -f *.log *.tmp study.pkl
	@echo "Clean up complete."
