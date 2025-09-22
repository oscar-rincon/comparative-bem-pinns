# Makefile to run all scripts in 03_comparison section
.PHONY: all run_03_comparison clean

# Define default target
all: run_03_comparison

# Target to run the full 03_comparison workflow
run_03_comparison:
	@echo "=============================================="
	@echo ">>> START: 03_comparison Workflow"
	@echo "=============================================="

	@echo "Running pinns_training_evaluation.py..."
	@python pinns_training_evaluation.py
	@echo "Finished pinns_training_evaluation.py."

	@echo "Running bem_solution.py..."
	@python bem_solution.py
	@echo "Finished bem_solution.py."

	@echo "Running comparison_bem.py..."
	@python comparison_bem.py
	@echo "Finished comparison_bem.py."

	@echo "Running comparison_pinns.py..."
	@python comparison_pinns.py
	@echo "Finished comparison_pinns.py."

	@echo "Running comparison_plot_top.py..."
	@python comparison_plot_top.py
	@echo "Finished comparison_plot_top.py."

	@echo "Running comparison_figure_compose.py..."
	@python comparison_figure_compose.py
	@echo "Finished comparison_figure_compose.py."

	@echo "Running sum_times.py..."
	@python sum_times.py
	@echo "Finished sum_times.py."

	@echo "=============================================="
	@echo ">>> END: 03_comparison Workflow"
	@echo "=============================================="
