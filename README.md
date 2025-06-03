# Robust analog quantum simulators by quantum error-detecting codes


## Overview

This repository provides code for numerical simulation of the robust analog quantum simulator scheme proposed in our [paper](https://arxiv.org/abs/2412.07764). The main goal is to demonstrate how encoding quantum many-body systems with quantum error-detecting codes passively can enhance evolution fidelity against 1-local coherent errors during analog quantum simulation. The code supports constructing encoded Hamiltonians, simulating their dynamics, and comparing the results with unencoded Hamiltonian simulations.

## Dependencies
- Python >= 3.8
- `numpy`
- `scipy`
- `matplotlib`
- This repo relies on the Python package `dynamite` for fast simulation of many-body quantum spin dynamics. To install `dynamite`, please follow the instructions in [https://github.com/GregDMeyer/dynamite](https://github.com/GregDMeyer/dynamite).


## File Descriptions

The folder `src/` contains the Python scipts used to numerically calculate the infidelity of the error-perturbed encoded/unencoded analog quantum simulation schemes for a variety of many-body spin Hamiltonians.
The file names are self-explanatory and follow the same naming convention.
Descriptions of the command line arguments can be found in the script files.

The folder `data/` contains the output `.txt` files when running the Python scripts in `src/`, as well as `.ipynb` notebooks for processing the data and generating the figures shown in the paper.
In particular, Fig. 3 and Fig. 5 in the arxiv paper can be found in the notebooks `data/plot.ipynb` and `data/plot_sweep_lamb.ipynb`, respectively.


