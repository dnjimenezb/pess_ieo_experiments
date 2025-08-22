"""Module containing functions used to solve different versions of the 0-1 knapsack problem
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from utils import get_predictions


def get_coef_zk(zk):
    """
    Get the variables indexes to generate no-good cuts

    :param zk:  0-1 knapsack solutions

    :return   list of indexes such that z[i] = 0/1
    """
    (NK, NI) = zk.shape
    idx_zero = dict()
    idx_one = dict()
    for k in range(NK):
        idx0 = []
        idx1 = []
        for i in range(NI):
            if zk[k, i] == 0:
                idx0.append(i)
            else:
                idx1.append(i)
        idx_zero[k] = idx0
        idx_one[k] = idx1

    return idx_zero, idx_one

def construct_base(weights, capacity):
    """
    Creates a Gurobi model to solve an instance of the 0-1 knapsack problem

    :param weights:     weights vector
    :param capacity:    knapsack capacity

    :return:  Gurobi model and variables
    """
    n_items = len(weights)
    model = gp.Model('Knapsack')
    model.setParam('OutputFlag', 0)
    z = model.addVars(n_items, vtype=GRB.BINARY)
    tot_weigh = gp.quicksum(z[j] * weights[j] for j in range(n_items))
    model.addConstr(tot_weigh <= capacity)

    return model, z, n_items

def get_solution(n_items, z_var):
    """
    Get the solution after solving an instance of the 0-1 knapsack problem

    :param n_items:   number of items
    :param z_var:     0-1 knapsack variables

    :return knapsack solution
    """
    solution = np.zeros(n_items)
    for j in range(n_items):
        solution[j] = z_var[j].X

    return solution

def solve_regular_knapsack(weights, capacity, costs):
    """
    Solve a regular instance of the 0-1 knapsack problem

    :param weights:      weights vector
    :param capacity:     knapsack capacity
    :param costs:        costs vector

    :return optimal objective value, optimal solution
    """
    model, z, n_items = construct_base(weights, capacity)
    tot_cost = gp.quicksum(z[j] * costs[j] for j in range(n_items))
    model.setObjective(tot_cost, GRB.MAXIMIZE)
    model.optimize()
    solution = get_solution(n_items, z)

    return model.objVal, solution

def solve_regular_no_good(weights, capacity, costs, z_k):
    """
    Solve a regular instance of the 0-1 knapsack problem including no-good cuts

    :param weights:      weights vector
    :param capacity:     knapsack capacity
    :param costs:        costs vector
    :param z_k:          knapsack solutions to be excluded

    :return optimal objective value, optimal solution
    """
    (n_k, _) = z_k.shape
    idx0, idx1 = get_coef_zk(z_k)
    model, z, n_items = construct_base(weights, capacity)
    tot_cost = gp.quicksum(z[j] * costs[j] for j in range(n_items))
    dk = model.addVars(n_k, vtype=GRB.BINARY)
    for k in range(n_k):
        model.addConstr(gp.quicksum(z[j] for j in idx0[k])
                             + gp.quicksum(1 - z[j] for j in idx1[k]) + dk[k] >= 1)
    violation = gp.quicksum(dk[k] * 1E3 for k in range(n_k))
    model.setObjective(tot_cost - violation, GRB.MAXIMIZE)
    model.optimize()
    solution = np.zeros(n_items)
    flag = False
    k_sol = 0
    for k in range(n_k):
        if dk[k].X == 1:
            flag = True
            k_sol = k
            break
    if flag:
        for j in range(n_items):
            solution[j] = z_k[k_sol, j]
    else:
        solution = get_solution(n_items, z)

    return tot_cost.getValue(), solution

def solve_worstcase(weights, capacity, costs, pred_cost, opt_sol_pred_cost, eps):
    """
    Solve an instance of a worst-case 0-1 knapsack problem, given a predicted objective value

    :param weights:             weights vector
    :param capacity:            knapsack capacity
    :param costs:               costs vector
    :param pred_cost:           predicted costs vector
    :param opt_sol_pred_cost:   optimal objective costs given the predicted costs
    :param eps:                 epsilon parameter used in equation (14c)

    :return: optimal objective value, optimal solution
    """
    model, z, n_items = construct_base(weights, capacity)
    pred_tot_cost = gp.quicksum(z[j] * pred_cost[j] for j in range(n_items))
    tot_cost = gp.quicksum(z[j] * costs[j] for j in range(n_items))
    model.addConstr(pred_tot_cost >= opt_sol_pred_cost - eps)
    model.setObjective(tot_cost, GRB.MINIMIZE)
    model.optimize()
    solution = get_solution(n_items, z)

    return model.objVal, solution

def get_bilevel_feasible_single(mp_sol, eps, costs, predictions, weights, capacity, type_c):
    """
    Generate a bilevel feasible solution, besides other knapsack feasible solutions used for cut generation
    for a single sample

    :param mp_sol:          master problem zs solution
    :param eps:             epsilon parameter used in equation (14c)
    :param costs:           knapsack costs vector
    :param predictions:     predicted costs vector
    :param weights:         weights vector
    :param capacity:        knapsack capacity
    :param type_c:          type of the cut (0 or 1) only used when using the branch-and-cut algorithm

    :return:    dictionary of solutions (including a bilevel feasible solution), bilevel feasible objective cost
    """
    n_items = len(costs)
    solutions = dict()
    opt_cost, dummy = solve_regular_knapsack(weights, capacity, costs)
    if type_c == 0:
        opt_sol_pc = gp.quicksum(mp_sol[j] * predictions[j] for j in range(n_items))
        dummy, solutions['WorstCase ZS-CUT'] = solve_worstcase(weights, capacity, costs, predictions, opt_sol_pc, eps)

    opt_obj, solutions['OPT Predicted Cost'] = solve_regular_knapsack(weights, capacity, predictions)
    feas_cost, solutions['Bilevel Feasible'] = solve_worstcase(weights, capacity, costs, predictions, opt_obj, eps)
    objective_value = feas_cost / opt_cost

    return solutions, objective_value

def get_bilevel_feasible(mp_sol, eps, costs, predictions, weights, capacity, type_c):
    """
    Generate a bilevel feasible solution, besides other knapsack feasible solutions used for cut generation
    for a set of samples

    :param mp_sol:          n_samples x n_items matrix of master problem zs solution
    :param eps:             epsilon parameter used in equation (14c)
    :param costs:           n_samples x n_items matrix of  knapsack costs
    :param predictions:     n_samples x n_items matrix of predicted costs
    :param weights:         weights vector
    :param capacity:        knapsack capacity
    :param type_c:          type of the cut (0 or 1) only used when using the branch-and-cut algorithm

    :return:    dictionary of solutions (including a bilevel feasible solution) for the set of samples,
                bilevel feasible objective cost for the set of samples
    """
    (n_scenarios, n_items) = costs.shape
    obj_val_array = np.zeros(n_scenarios)
    sol_stack = dict()

    for i in range(n_scenarios):
        sol_i, obj_val_array[i] = get_bilevel_feasible_single(mp_sol[i, :], eps, costs[i, :], predictions[i, :], weights, capacity, type_c)
        for name in sol_i.keys():
            if i == 0:
                sol_stack[name] = sol_i[name]
            else:
                sol_stack[name] = np.vstack((sol_stack[name], sol_i[name]))

    objective_value = np.sum(obj_val_array) / n_scenarios

    return sol_stack, objective_value

def get_objective_value(b_tested, ns_train, weights, capacity, costs, features):
    """
    Compute the objective value (loss) for a given set of samples, for a given b_tested predictor

    :param b_tested:    n_items x n_features matrix of the predictor
    :param ns_train:    number of training samples
    :param weights:     weights vector
    :param capacity:    knapsack capacity
    :param costs:       n_samples x n_items matrix of master problem zs solution
    :param features:    n_samples x n_features matrix of features

    :return:    objective value (loss)
    """
    (n_items, _) = b_tested.shape
    predictions = np.zeros((ns_train, n_items))
    for i in range(ns_train):
        predictions[i, :] = get_predictions(b_tested, features[i, :])

    _, objective_value = get_bilevel_feasible(mp_sol=np.zeros((ns_train, n_items)), eps=0, costs=costs,
                                              predictions=predictions, weights=weights, capacity=capacity, type_c=1)

    return objective_value

def generate_opt_points(n_points, ns_train, n_items, weights, capacity, costs):
    """
    Generate a set of feasible solutions with decreasing objective value (the first one is the optimal solution)

    :param n_points:    number of solutions
    :param ns_train:    number of training samples
    :param n_items:     number of knapsack items
    :param weights:     weights vector
    :param capacity:    knapsack capacity
    :param costs:       n_samples x n_items matrix of knapsack costs

    :return: dictionary containing feasible solutions
    """
    z_gen = np.zeros((n_points, ns_train, n_items))
    zk_dict = dict()
    for k in range(n_points):
        for i in range(ns_train):
            _, z_gen[k, i, :] = solve_regular_no_good(weights, capacity, costs[i, :], z_gen[:, i, :])
            zk_dict[(i, k)] = z_gen[k, i, :]

    return zk_dict

def calc_opt_in_sample(ns_train, n_items, weights, capacity, costs):
    """
    Calculate optimal in-sample solutions

    :param ns_train:    number of training samples
    :param n_items:     number of knapsack items
    :param weights:     weights vector
    :param capacity:    knapsack capacity
    :param costs:       n_samples x n_items matrix of knapsack costs

    :return:    optimal in-sample costs, optimal solution solutions
    """
    opt_is_cost = np.zeros(ns_train)
    opt_is_sol = np.zeros((ns_train, n_items))
    for i in range(ns_train):
        opt_is_cost[i], opt_is_sol[i, :] = solve_regular_knapsack(weights, capacity, costs[i, :])

    return opt_is_cost, opt_is_sol
