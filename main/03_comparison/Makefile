# Makefile to run Python scripts in sequence

.PHONY: all run_pinns_training run_bem_solution_pinns_evaluation run_comparison_bem run_comparison_pinns run_comparison_plot_top run_comparison_plot_bottom clean

# Define default target
all: run_pinns_training run_bem_solution_pinns_evaluation run_comparison_bem run_comparison_pinns run_comparison_plot_top run_comparison_plot_bottom

# Target to run pinns_training.py
run_pinns_training:
	@echo "Running pinns_training.py..."
	@python pinns_training.py
	@echo "Finished pinns_training.py..."

# Target to run bem_solution_pinns_evaluation.py
run_bem_solution_pinns_evaluation:
	@echo "Running bem_solution_pinns_evaluation.py..."
	@python bem_solution_pinns_evaluation.py
	@echo "Finished bem_solution_pinns_evaluation.py..."

# Target to run comparison_bem.py
run_comparison_bem:
	@echo "Running comparison_bem.py..."
	@python comparison_bem.py
	@echo "Finished comparison_bem.py..."

# Target to run comparison_pinns.py
run_comparison_pinns:
	@echo "Running comparison_pinns.py..."
	@python comparison_pinns.py
	@echo "Finished comparison_pinns.py..."	

# Target to run comparison_plot_top.py
run_comparison_plot_top:
	@echo "Running comparison_plot_top.py..."
	@python comparison_plot_top.py
	@echo "Finished comparison_plot_top.py..."	

# Target to run comparison_plot_bottom.py
run_comparison_plot_bottom:
	@echo "Running comparison_plot_bottom.py..."
	@python comparison_plot_bottom.py
	@echo "Finished comparison_plot_bottom.py..."	

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* data/*
	@echo "Clean up complete."
