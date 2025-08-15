# Makefile to run Python scripts in sequence

.PHONY: all clean run_comparison_bem run_comparison_pinns run_comparison_plot_a run_comparison_plot_b

# Define default target
all: clean run_comparison_bem run_comparison_pinns run_comparison_plot_a run_comparison_plot_b

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

# Target to run comparison_plot_a.py
run_comparison_plot_a:
	@echo "Running comparison_plot_a.py..."
	@python comparison_plot_a.py
	@echo "Finished comparison_plot_a.py..."	

# Target to run comparison_plot_b.py
run_comparison_plot_b:
	@echo "Running comparison_plot_b.py..."
	@python comparison_plot_b.py
	@echo "Finished comparison_plot_b.py..."	

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/*  data/*
	@echo "Clean up complete."	