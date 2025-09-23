# Makefile to run all main scripts in sequence by section
.PHONY: all clean \
        run_01_analytical_solution \
        run_02_hyperparameter_optimization \
        run_03_comparison \
        run_04_generalization

# --------------------------------------------------------------------
# Default target
all: run_01_analytical_solution run_02_hyperparameter_optimization run_03_comparison run_04_generalization

# --------------------------------------------------------------------
# 01 Analytical solution
run_01_analytical_solution:
	@echo "=============================================="
	@echo ">>> START: 01 Analytical Solution"
	@echo "=============================================="
	@python main/01_analytical_solution/analytical_solution.py
	@echo "=============================================="
	@echo ">>> END: 01 Analytical Solution"
	@echo "=============================================="

# --------------------------------------------------------------------
# 02 Hyperparameter optimization
run_02_hyperparameter_optimization:
	@echo "=============================================="
	@echo ">>> START: 02 Hyperparameter Optimization"
	@echo "=============================================="
	@python main/02_hyperparameter_optimization/optimize_pinns.py
	@python main/02_hyperparameter_optimization/training_optimized_pinns.py
	@python main/02_hyperparameter_optimization/plot_optuna_results.py
	@python sum_times.py
	@echo "=============================================="
	@echo ">>> END: 02 Hyperparameter Optimization"
	@echo "=============================================="

# --------------------------------------------------------------------
# 03 Comparison
run_03_comparison:
	@echo "=============================================="
	@echo ">>> START: 03 Comparison"
	@echo "=============================================="
	@python main/03_comparison/pinns_training_evaluation.py
	@python main/03_comparison/bem_solution_evaluation.py
	@python main/03_comparison/comparison_bem.py
	@python main/03_comparison/comparison_pinns.py
	@python main/03_comparison/comparison_plot_time_error.py
	@python main/03_comparison/comparison_figure_compose.py
	@python sum_times.py
	@echo "=============================================="
	@echo ">>> END: 03 Comparison"
	@echo "=============================================="

# --------------------------------------------------------------------
# 04 Generalization
run_04_generalization:
	@echo "=============================================="
	@echo ">>> START: 04 Generalization"
	@echo "=============================================="
	@python main/04_generalization/generalization_bem.py
	@python main/04_generalization/generalization_pinns.py
	@python main/04_generalization/generalization_figure_composition.py
	@python sum_times.py
	@echo "=============================================="
	@echo ">>> END: 04 Generalization"
	@echo "=============================================="
