# Makefile to run all PINN scripts in sequence

.PHONY: run_02_hyperparameter_optimization

run_02_hyperparameter_optimization:
	@echo "=============================================="
	@echo ">>> START: Hyperparameter Optimization Workflow"
	@echo "=============================================="

	@echo "Running optimize_pinns.py with Optuna..."
	@python3 optimize_pinns.py
	@echo "Finished optimize_pinns.py."

	@echo "Running trainning_optimized_pinns.py..."
	@python3 trainning_optimized_pinns.py
	@echo "Finished trainning_optimized_pinns.py."

	@echo "Running plot_optuna_results.py..."
	@python3 plot_optuna_results.py
	@echo "Finished plot_optuna_results.py."

	@echo "=============================================="
	@echo ">>> END: Hyperparameter Optimization Workflow"
	@echo "=============================================="
