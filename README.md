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
combinatorial optimization problems. Our method solves an  $\varepsilon$-approximation of the 
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

Experiments consist of three main parts: computing IEO predictors, performing out-of-sample testing, and displaying the desired tables and charts.  
All output files are saved in the `output` folder.

⚠️ **Important**: The experiments are designed to be executed **sequentially**. In particular, the IEO matrices must be computed before running out-of-sample tests, 
as the latter depend on the former. While parallelization can be used to solve individual instances, full experiment pipelines (e.g., reproducing tables and figures) require that all intermediate results be available beforehand.

### Computing IEO predictors

Experiments can be run from the root folder by executing the script `solve_ieo.py` using the commands listed below.  
The script computes the $\mathbf{W}$ matrices described in Section 4 for each of the tested methods. These matrices are saved in the `b_matrices` subfolder, where additional subfolders indicate the number of knapsack items and the number of training samples used.

Please note that running all the experiments presented in the paper may take a considerable amount of time. This script also computes linear regression and SPO+ predictors.

The main parameters of the script `solve_ieo.py` are the following.

#### Parameters
- `--n_items`: Number of items in the knapsack instance. Choose 10 or 20.
- `--n_samples`: Number of samples used in the predictor computation. Choose 100, 200, 300, 400, or 500 (only for the case with 10 items). Several values can be passed.
- `--method`: Method used to solve the pessimistic IEO computation. Select either `bc` (branch-and-cut) or `ccg` (column-and-constraint generation).
- `--timelimit`: Maximum runtime (in seconds) for each instance.
- `--instances`: IDs of the instances to be executed. Choose from 0 to 10 (default: `[0, 1, ..., 10]`).
- `--delta`: Values of $\delta$ to be used. Choose 1, 3, or 5 (default: `[1, 3, 5]`).

📌 Example

Perform the pessimistic IEO predictor computation for all knapsack instances with 10 items, using training sample sizes from 100 to 500, the branch-and-cut algorithm, and a time limit of 600 seconds.

```bash
python ./python/solve_ieo.py --n_items 10 --n_samples 100 200 300 400 500 --method bc --timelimit 600

```

### Performing out-of-sample tests

Out-of-sample experiments can be run from the root folder by executing the script `out_of_sample_tests.py`. This script performs two tasks.

First, it solves `n_test` knapsack instances using out-of-sample data and computes the out-of-sample loss for each tested method. These results are saved in the `out_of_sample_results` subfolder, 
where additional subfolders indicate the number of knapsack items and the number of training samples used.

Second, the outputs of the previous step are used to compute all statistical indicators reported in the manuscript. These results are saved in a file called `final_stats.csv`.

The main parameters of the script `out_of_sample_tests.py` are the following.

#### Parameters
- `--n_items`: Number of items in the knapsack instance. Choose 10 or 20.
- `--ns_train`: Number of samples used in the predictor computation. Choose 100, 200, 300, 400, or 500 (only for the case with 10 items). Several values can be passed.
- `--ns_test`: Number of samples used in the out-of-sample testing.
- `--instances`: IDs of the instances to be executed. Choose from 0 to 10 (default: [0, 1, ..., 10]).
- `--delta`: Values of $\delta$ to be used. Choose between 1, 3 or 5 (default: [1, 3, 5]).

Note that the $\mathbf{W}$ matrices must be computed beforehand for correct execution.

📌 Example

Perform out-of-sample tests for all knapsack instances with 10 items, for training sample sizes from 100 to 500, using 1000 test samples.

```bash
python ./python/out_of_sample_tests.py --n_items 10 --ns_train 100 200 300 400 500 --ns_test 1000
```

### Displaying outputs

This section provides specific instructions for displaying the outputs reported in the manuscript, namely Tables 1–3 and Figures 1–2.

Since out-of-sample testing relies on random sampling, numerical results may vary slightly across runs. Nevertheless, the reported trends, 
observations, and conclusions are preserved. Moreover, as fixed time limits are used, computational performance may vary depending on the machine 
on which the experiments are executed.

#### Table 1

To reproduce Table 1, IEO matrices must first be computed for instances 1 and 10, time limits of 600, 1200, and 1800 seconds, training sample 
sizes of 100, 300, and 500, and both tested methods (branch-and-cut and column-and-constraint generation). This can be done by executing:

```bash
python ./python/solve_ieo.py --n_items 10 --n_samples 100 300 500 --instances 1 10 --delta 3 --table_1 True --method bc --timelimit 600 1200 1800
python ./python/solve_ieo.py --n_items 10 --n_samples 100 300 500 --instances 1 10 --delta 3 --table_1 True --method ccg --timelimit 600 1200 1800
```

The output files are saved in the folder `output/table_1_res`. The table can then be displayed as a Pandas DataFrame by executing:

```bash
python ./python/gen_table_1.py
```

#### Tables 2 and 3

To reproduce Tables 2 and 3, IEO matrices must first be computed for instances 0, 5, and 7 and training sample sizes of 100 and 200, using the 
time limit specified in the manuscript (1800 seconds). Out-of-sample experiments must then be performed using 1000 test samples. This can be done 
by executing:

```bash
python ./python/solve_ieo.py --n_items 10 --n_samples 100 200 --instances 0 5 7 --table_23 True --method bc --timelimit 1800
python ./python/out_of_sample_tests.py --n_items 10 --ns_train 100 200 --instances 0 5 7 --table_23 True --ns_test 1000
```

The output files are saved in the folder `output/table_23_res`, where the subfolders `b_matrices` and `out_of_sample_results` store 
the $\mathbf{W}$ matrices and the corresponding out-of-sample losses $\bar{\ell}_{IEO}$, respectively. A summary file named 
`final_stats.csv` is also generated.

Both tables can then be displayed as Pandas DataFrames by executing:

```bash
python ./python/gen_tables_23.py
```

#### Figures 1 and 2

These experiments are the most computationally intensive. As in the previous cases, IEO matrices must be computed first, followed 
by out-of-sample testing.

To reproduce Figure 1, execute:

```bash
python ./python/solve_ieo.py --n_items 10 --n_samples 100 200 300 400 500 --method bc --timelimit 1800
python ./python/out_of_sample_tests.py --n_items 10 --ns_train 100 200 300 400 500 --ns_test 1000
```

The output files are saved in the folder `output/figures_12_res`, where the subfolders `b_matrices` and `out_of_sample_results` 
store the $\mathbf{W}$ matrices and the out-of-sample losses $\bar{\ell}_{IEO}$, respectively. 
A summary file final_stats.csv is also generated.

Figure 1 can then be displayed by executing:

```bash
python ./python/gen_plot.py --n_items 10
```

For Figure 2, the procedure is identical. To reproduce it, execute:

```bash
python ./python/solve_ieo.py --n_items 20 --n_samples 100 200 300 400 --method bc --timelimit 1800
python ./python/out_of_sample_tests.py --n_items 20 --ns_train 100 200 300 400 --ns_test 1000
python ./python/gen_plot.py --n_items 20
```

### Notes
- Some method-instance combinations are intentionally skipped due to known limitations (e.g., SPO+ with convex hull inequalities for certain instances). These are not errors.
- Previous versions of the code displayed messages for skipped instances; these have been removed to avoid confusion.
