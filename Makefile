# Makefile to run Python scripts in sequence

.PHONY: all clean run_01_analytical_solution 

# Define default target
all: run_01_analytical_solution

# Target to run Schrodinger_main.py
run_01_analytical_solution:
	@echo "Running 01_analytical_solution.py..."
	@python main/01_analytical_solution/analytical_solution.py
	@echo "Finished 01_analytical_solution.py..."