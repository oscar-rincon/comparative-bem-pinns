# Comparison of Accuracy and Computational Efficiency Between Methods

![errors](figures/errors.svg)


## 🚀 How to Run

To execute the full workflow (**BEM → PINNs → Plots**), open a terminal in the project directory and run:

```bash
make all
```

Run the BEM script:

```bash
make run_comparison_bem
```

Run the PINNs script:

```bash
make run_comparison_pinns
```

Run the comparison plot a:

```bash
make run_comparison_plot_a
```

Generate the comparison plot b:

```bash
make run_comparison_plot_b
```

## 🧹 Cleaning Up

```bash
make clean
```