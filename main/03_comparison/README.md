# Comparison of Accuracy and Computational Efficiency Between Methods

![comparison](figures/comparison.svg)


##  How to Run

To execute the full workflow (**PINNs Training → BEM & PINNs Evaluation → Comparisons → Plots**), open a terminal in the project directory and run:

```bash
make all
```

Run the PINNs training script:

```bash
make run_pinns_training
```

The aproximate time required is around 15 minutes using the stored models.

```bash
make run_bem_solution_pinns_evaluation
```

Run the BEM script:

```bash
make run_comparison_bem
```

Run the PINNs script:

```bash
make run_comparison_pinns
```

Run the comparison plot top:

```bash
make run_comparison_plot_top
```

Generate the comparison plot bottom:

```bash
make run_comparison_plot_bottom
```

## Estimated time

The execution of this script takes approximately 241.72 s.