""" Module containing functions to read/generate data used in the experiments.
Functions to compute linear regression and SPO+ predictors are also included.
"""
import pickle
from sklearn.linear_model import LinearRegression
from utils import *
from single_knapsack import *
from spo_plus import solve_spo_plus
from spo_plus_ch import solve_spo_plus_ch

np.random.seed(1234)

def read_data(path_data, ns_train, mode=''):
    """
    Read a pickle file and extract data from it.

    :param path_data: path to pickle file
    :param ns_train:  number of training samples
    :param mode:      either 'train' or 'test'

    :return: instances data (depending on the mode selected)
    """
    with open(path_data, 'rb') as f:
        data = pickle.load(f)

    n_items = data['data']['m']
    n_features = data['data']['k']
    deg = data['data']['delta']
    eps = data['data']['eps']
    weights = data['knapsack']['weights']
    capacity = data['knapsack']['capacity']
    b_gt = data['data']['V']
    raw_cost = data['data']['C'].transpose()[0:ns_train, :]
    features = data['data']['W'].transpose()[0:ns_train, :]
    costs = raw_cost / np.amax(raw_cost)

    if mode == 'train':
        return n_items, n_features, features, costs, weights, capacity

    elif mode == 'test':
        return n_items, n_features, b_gt, deg, eps, weights, capacity

def gen_test_samples(ns_test, n_items, n_features, b_gt, deg, eps):
    """
    Create samples of features and knapsack costs for out-of-sample testing

    :param ns_test:     number of test samples
    :param n_items:     number of knapsack items
    :param n_features:  number of features
    :param b_gt:        ground truth b matrix
    :param deg:         delta parameter using in the testing data generation (section 4.1)
    :param eps:         alpha parameter using in the testing data generation (section 4.1)

    :return:     features testing samples, costs testing samples
    """
    features_test = np.ones((ns_test, n_features))
    costs_test = np.zeros((ns_test, n_items))
    features_test[:, :n_features - 1] = np.random.uniform(-1, 1, (ns_test, n_features - 1))
    eps = np.random.uniform(1 - eps, 1 + eps, (ns_test, n_items))
    features_test[:, n_features - 1] = np.ones(ns_test)
    for i in range(ns_test):
        for j in range(n_items):
            base_value = np.dot(b_gt[j, :], features_test[i, :])
            costs_test[i, j] = eps[i, j] * base_value ** deg + 0.5 * (np.random.exponential(scale=1) - 1)

    costs_test = costs_test / np.amax(costs_test)

    return features_test, costs_test

def calc_b_lr(n_items, n_features, features, costs):
    """
    Compute the predictor of knapsack costs using linear regression

    :param n_items:     number of knapsack items
    :param n_features:  number of features
    :param features:    n_samples x n_features matrix of features
    :param costs:       n_samples x n_items matrix of costs

    :return: linear regression predictor matrix
    """
    b_lr = np.zeros((n_items, n_features))
    for j in range(n_items):
        Xj = features[:, :n_features - 1]
        yj = costs[:, j]
        reg = LinearRegression().fit(Xj, yj)
        b_lr[j, :n_features - 1] = reg.coef_
        b_lr[j, n_features - 1] = reg.intercept_

    return b_lr

def calc_spop(weights, capacity, costs, features, opt_is_sol):
    """
    Compute the predictor of knapsack costs using the SPO+ method

    :param weights:     vector of knapsack weights
    :param capacity:    knapsack capacity
    :param costs:       n_samples x n_items matrix of costs
    :param features:    n_samples x n_features matrix of features
    :param opt_is_sol:  n_samples x n_items matrix of optimal in-sample solutions

    :return: SPO+ predictor matrix
    """
    b_spo = solve_spo_plus(feat_matrix=features, cost_matrix=costs, weights=weights, capacity=capacity,
                           opt_matrix=opt_is_sol)

    return b_spo

def calc_spop_ch(path_coefficients, n_items, costs, features, opt_is_sol):
    """
        Compute the predictor of knapsack costs using the SPO+ method enhanced with the convex hull description
        of the knapsack polytope

        :param path_coefficients:   path to the file containing the convex hull inequality coefficients
        :param n_items:             number of knapsack items
        :param costs:               n_samples x n_items matrix of costs
        :param features:            n_samples x n_features matrix of features
        :param opt_is_sol:          n_samples x n_items matrix of optimal in-sample solutions

        :return: SPO+ convex hull enhanced predictor matrix
        """
    df_coef = pd.read_csv(path_coefficients)
    coef_matrix = df_coef[[str(x) for x in range(1, n_items + 1)]].values
    rhs_vector = df_coef['rhs'].values
    b_spo_ch = solve_spo_plus_ch(feat_matrix=features, cost_matrix=costs, coef_matrix=coef_matrix,
                                 rhs_vector=rhs_vector, opt_matrix=opt_is_sol)

    return b_spo_ch
