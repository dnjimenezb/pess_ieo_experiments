# pess_ieo_experiments

This repository is by Diego Jiménez and contains the Python source code to reproduce the
experiments of the paper entitled "[Pessimistic bilevel optimization approach for decision-focused learning](https://arxiv.org/abs/2501.16826)", 
written jointly with Bernardo K. Pagnoncelli and Hande Yaman.

If you find this repository useful, please consider citing our publication.

---

## Introduction

The recent interest in contextual optimization problems, where randomness is associated with 
side information, has led to two primary strategies for formulation and solution. The first, 
estimate-then-optimize, separates the estimation of the problem's parameters from the 
optimization process. The second, decision-focused optimization, integrates the optimization 
problem's structure directly into the prediction procedure. In this work, we propose a 
pessimistic bilevel approach for solving general decision-focused formulations of 
combinatorial optimization problems. Our method solves an  $\epsilon$-approximation of the 
pessimistic bilevel problem using a specialized cut generation algorithm. We benchmark its 
performance on the 0-1 knapsack problem against estimate-then-optimize and decision-focused 
methods, including the popular SPO+ approach. Computational experiments highlight the proposed 
method's advantages, particularly in reducing out-of-sample regret. 

---

## Dependencies

### Requirements
- Python 3.10+ 
- The required dependencies are listed in `requirements.txt`. From the root folder, install them with

```bash
pip install -r requirements.txt
```

---

## Instructions

Experiments consist of two main parts: computing IEO predictors and performing out-of-sample testing.

### Computing IEO predictors

Experiments can be run from the root folder by executing the script `solve_ieo.py`, using the commands listed below. Please note that running all 
the experiments presented in the paper may take a considerable amount of time. Running this script also computes linear regression and SPO+ predictors.

#### Parameters
- `--n_items`: Number of items in the knapsack instance. Choose 10 or 20.
- `--n_samples`: Number of samples used in the predictor computation. Choose 100, 200, 300, 400 or 500 (only for the case with 10 items). Several values can be passed.
- `--method`: Method used to solve the pessimistic IEO computation. Select either bc (branch-and-cut) or ccg (column-and-constraint generation).
- `--timelimit`: Maximum runtime (in seconds) for each instance.

📌 Example

Perform the pessimistic IEO predictor computation for the knapsack instances with 10 items, from 100 to 500 training samples, 
using the branch-and-cut algorithm with a time limit of 1000 seconds.

```bash
python ./python/solve_ieo.py --n_items 10 --n_samples 100 200 300 400 500 --method bc --timelimit 1000
```

### Performing out-of-sample tests

Experiments can be run from the root folder by executing the script `out_of_sample_tests.py`, using the commands listed below. 
Notice that predictors should be already computed and saved in the folder `output\b_matrices`, for the desired values of
knapsack items and training samples. To generate the boxplots charts, use `gen_plot.py`.

#### Parameters
- `--n_items`: Number of items in the knapsack instance. Choose 10 or 20.
- `--ns_train`: Number of samples used in the predictor computation. Choose 100, 200, 300, 400 or 500 (only for the case with 10 items). Several values can be passed.
- `--ns_test`: Number of samples to be used in the out-of-sample testing.

📌 Example

Perform out-of-sample tests for the knapsack instances with 10 items, for the 100 - 500 training samples cases, using 1000 testing samples.
Then, generate the boxplot chart for the previous case.

```bash
python ./python/out_of_sample_tests.py --n_items 10 --ns_train 100 200 300 400 500 --ns_test 1000
python ./python/gen_plot.py --n_items 10
```
