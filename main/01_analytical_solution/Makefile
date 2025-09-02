# Makefile to run Python scripts in sequence

.PHONY: all run_analytical_solution 

# Define default target
all: run_analytical_solution

# Target to run analytical_solution.py
run_analytical_solution:
	@echo "Running analytical_solution.py..."
	@python analytical_solution.py
	@echo "Finished analytical_solution.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* 
	@echo "Clean up complete."		