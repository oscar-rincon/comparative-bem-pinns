# Makefile to run all scripts in 04_generalization section
.PHONY: all run_04_generalization

# Define default target
all: run_04_generalization

# Target to run the full 04_generalization workflow
run_04_generalization:
	@echo "=============================================="
	@echo ">>> START: 04_generalization Workflow"
	@echo "=============================================="

	@echo "Running generalization_bem.py..."
	@python generalization_bem.py
	@echo "Finished generalization_bem.py."

	@echo "Running generalization_pinns.py..."
	@python generalization_pinns.py
	@echo "Finished generalization_pinns.py."

	@echo "Running generalization_plot.py..."
	@python generalization_plot.py
	@echo "Finished generalization_plot.py."

	@echo "=============================================="
	@echo ">>> END: 04_generalization Workflow"
	@echo "=============================================="
