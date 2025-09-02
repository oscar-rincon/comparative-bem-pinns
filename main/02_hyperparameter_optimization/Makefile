# Makefile to run Optuna minimization script
 
.PHONY: all optimize plot_optimize

all: optimize plot_optimize

# Run hyperparameter optimization
optimize:
	@echo "Running optimize_pinns.py with Optuna..."
	@python3 optimize_pinns.py
	@echo "Finished optimize_pinns.py..."

# Run plotting script
plot_optimize:
	@echo "Running plot_optuna_results.py..."
	@python3 plot_optuna_results.py
	@echo "Finished plot_optuna_results.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* || true
	@rm -rf __pycache__ || true
	@rm -f *.log *.tmp study.pkl
	@echo "Clean up complete."
