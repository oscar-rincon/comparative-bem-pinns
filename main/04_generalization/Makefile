# Makefile to run Python scripts in sequence

.PHONY: all clean run_comparison_bem run_comparison_pinns run_comparison_bem_pinns  run_comparison_plot

# Define default target
all: clean run_generalization_bem run_generalization_pinns run_generalization_plot

# Target to run generalization_bem.py
run_generalization_bem:
	@echo "Running generalization_bem.py..."
	@python generalization_bem.py
	@echo "Finished generalization_bem.py..."

# Target to run generalization_pinns.py
run_generalization_pinns:
	@echo "Running generalization_pinns.py..."
	@python generalization_pinns.py
	@echo "Finished generalization_pinns.py..."
 
# Target to run generalization_bem_pinns_plot.py
run_generalization_plot:
	@echo "Running generalization_plot.py..."
	@python generalization_plot.py
	@echo "Finished generalization_plot.py..."

