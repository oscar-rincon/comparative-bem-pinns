# Comparative Analysis of Wave Scattering Numerical Modeling Using the Boundary Element Method and Physics-Informed Neural Networks

Here, we present a comparative study between PINNs and BEM for solving the two-dimensional Helmholtz equation, focusing on the problem of wave scattering. We evaluate both methods in terms of accuracy, computational efficiency, and generalization, highlighting their respective strengths and limitations.

## Installation or required Python packages

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda env create -f environment.yml
conda activate comparative-pinns-bem-env
```

To verify the packages installed in your `comparative-pinns-bem-env` conda environment, you can use the following command:

```
conda list -n comparative-pinns-bem-env
```

## Repository Organisation

`main/`:

- `01_analytical_solution/`: Results in Figure 4. Analytical estimation of scattering.

- `02_hyperparameter_optimization/`: Results in Figure 5.

- `03_comparison/`: Results in Figure 6. Performance evaluation of BEM and PINNs.

- `04_generalization/`: Results in Figure 7. Scattered field computed by BEM and PINNs outside the training region.

## Scripts execution order