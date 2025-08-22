"""Module to compute the SPO+ predictor for the continuous knapsack problem
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_spo_plus(feat_matrix, cost_matrix, weights, capacity, opt_matrix):
    """
    :param feat_matrix:     n_samples x n_features matrix of features
    :param cost_matrix:     n_samples x n_items matrix of costs
    :param weights:         weights vector
    :param capacity:        knapsack capacity
    :param opt_matrix:      n_samples x n_items matrix of optimal in-sample solutions

    :return: SPO+ predictor
    """
    (NS, NF) = feat_matrix.shape
    NI = len(weights)
    model = gp.Model('SPOp_Knapsack')
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 500)
    B = model.addVars(NI, NF, vtype=GRB.CONTINUOUS)
    d1 = model.addVars(NS, vtype=GRB.CONTINUOUS, lb=0.0)
    d2 = model.addVars(NS, NI, vtype=GRB.CONTINUOUS, lb=0.0)
    sample_cost = dict()
    sum_d1 = dict()
    sum_d1_constr = dict()
    sum_d2 = dict()
    rhs = dict()
    for i in range(NS):
        sample_cost[i] = gp.quicksum((gp.quicksum(B[j, l] * feat_matrix[i, l] for l in range(NF)))
                                     * opt_matrix[i, j] for j in range(NI))
        sum_d1[i] = d1[i] * capacity
        sum_d2[i] = gp.quicksum(d2[i, j] for j in range(NI))
        for j in range(NI):
            rhs[(i, j)] = (2 * (gp.quicksum(B[j, l] * feat_matrix[i, l] for l in range(NF)))
                           - cost_matrix[i, j])
            sum_d1_constr[(i, j)] = d1[i] * weights[j]

    tot_cost = gp.quicksum(sum_d1[i] + sum_d2[i] - 2 * sample_cost[i] for i in range(NS))
    model.setObjective(tot_cost / NS, GRB.MINIMIZE)

    for i in range(NS):
        for j in range(NI):
            model.addConstr(sum_d1_constr[(i, j)] + d2[i, j] >= rhs[(i, j)])

    model.optimize()
    Bsol = np.zeros((NI, NF))
    for j in range(NI):
        for l in range(NF):
            Bsol[j, l] = B[j, l].X

    return Bsol
