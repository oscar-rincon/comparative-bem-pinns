# Makefile to run Python scripts in sequence

.PHONY: all clean run_comparison_bem run_comparison_pinns run_comparison_bem_pinns  run_comparison_plot

# Define default target
all: clean run_comparison_bem run_comparison_pinns run_comparison_bem_pinns  run_comparison_plot

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

# Target to run comparison_bem_pinns.py
run_comparison_bem_pinns:
	@echo "Running comparison_bem_pinns.py..."
	@python comparison_bem_pinns.py
	@echo "Finished comparison_bem_pinns.py..."	

# Target to run comparison_bem_pinns_plot.py
run_comparison_plot:
	@echo "Running comparison_plot.py..."
	@python comparison_plot.py
	@echo "Finished comparison_plot.py..."	

