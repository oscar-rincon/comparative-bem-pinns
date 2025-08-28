# Makefile to run Optuna minimization script

.PHONY: all clean run_hyperparameter_optimization

# Define default target
all: run_hyperparameter_optimization

# Target to run hyperparameter optimization
run_hyperparameter_optimization:
	@echo "Running objective_function_minimization_plot.py with Optuna..."
	@python objective_function_minimization_plot.py
	@echo "Finished objective_function_minimization_plot.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* 
	@rm -rf __pycache__ 
	@rm -rf *.log *.tmp
	@echo "Clean up complete."
