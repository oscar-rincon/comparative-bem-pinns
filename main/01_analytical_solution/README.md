# Analytical Solution

![displacement_exact](figures/displacement_exact.svg)


## 🚀 How to Run

To execute the full workflow (**BEM → PINNs → Plot**), open a terminal in the project directory and run:

```bash
make all
```

Run only the BEM script:

```bash
make run_generalization_bem
```

Run only the PINNs script:

```bash
make run_generalization_pinns
```

Generate only the comparison plot:

```bash
make run_generalization_plot
```

## 🧹 Cleaning Up

```bash
make clean
```