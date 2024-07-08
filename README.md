# Reproducing Results for "Regression-Based Law of Energy Efficiency in Wireless Sensor Networks"

**Authors:** F. O. C. Prado and M. Brigato

This repository contains the code to reproduce the results of the paper "Regression-Based Law of Energy Efficiency in Wireless Sensor Networks."

## Notebooks and Scripts

### Transmitted_Power.ipynb
Use the `Transmitted_Power.ipynb` notebook to understand how all transmitted powers used in the subsequent regression analysis were obtained. Here you find the first two plots of the paper. 

### power.py
The `power.py` file includes two functions:
- `kmeans()`: Generates transmitted powers using the k-means clustering algorithm.
- `grid()`: Generates transmitted powers using a grid-based method.

These functions are based on the code in `Transmitted_Power.ipynb` and reproduce the same results.

### Regression_Analysis.ipynb
Use the `Regression_Analysis.ipynb` notebook to understand how the regression results and plots used in the paper were obtaine

## Getting Started

1. To run all codes properly we recommend to clone this repository:
   ```bash
   git clone https://github.com/mbiot2022/Regression_energetic_efficiency.git


