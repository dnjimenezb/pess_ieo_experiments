"""Module to solve the pessimistic IEO for the 0-1 knapsack using column-and-constraint generation

This module solves the pessimistic IEO for the 0-1 knapsack using column-and-constraint generation
This implementation uses the Gurobi MILP solver to solve the master and sub-problems.
"""

import time
from utils import *
from single_knapsack import *


class KnapsackCCG:
    def __init__(self, features, costs, opt_cost, init_sol, weights, capacity, b_fix, eps, bigM,
                       b_min, b_max, beta, zk_initial, n_try, alpha, time_limit, max_iter, gap_tol):
        """
        :param features:        n_samples x n_features features matrix
        :param costs:           n_samples x n_items costs matrix
        :param opt_cost:        vector of optimal costs for each sample
        :param init_sol:        initial variable values used for the warm-start
        :param weights:         vector of knapsack items weights
        :param capacity:        knapsack capacity
        :param b_fix:           b value used in the initialization
        :param eps:             epsilon parameter used in equation (14c)
        :param bigM:            big-M value used in equation (14c)
        :param b_min:           minimum b value allowed
        :param b_max:           maximum b value allowed
        :param beta:            lasso regularization weight
        :param zk_initial:      dictionary containing initial extreme points of the knapsack polytope for each sample
        :param n_try:           number of iterations used for the cut generation (section 3.4.4)
        :param alpha:           parameter used in the cut generation (section 3.4.4)
        :param time_limit:      time limit for the column-and-constraint generation algorithm
        :param max_iter:        maximum number of iterations used for the column-and-constraint generation algorithm
        :param gap_tol:         optimality gap tolerance
        """
        (self.n_samples, self.n_items) = np.shape(costs)
        (_, self.n_features) = np.shape(features)
        self.features = features
        self.costs = costs
        self.opt_cost = opt_cost
        self.init_sol = init_sol
        self.weights = weights
        self.capacity = capacity
        self.b_fix = b_fix
        self.eps = eps
        self.bigM = bigM
        self.b_min = b_min
        self.b_max = b_max
        self.beta = beta
        self.zk = zk_initial
        self.n_try = n_try
        self.alpha = alpha
        self.time_limit = time_limit
        self.max_iter = max_iter
        self.gap_tol = gap_tol
        n_init = list(zk_initial.keys())[-1][1] + 1
        self.zk_np = n_init * np.ones(self.n_samples)
        self.b_sol = np.zeros((self.n_items, self.n_features))
        self.cb_sol = np.zeros((self.n_items, self.n_features))
        self.mp_sol = np.zeros((self.n_samples, self.n_items))
        self.predictions = np.zeros((self.n_samples, self.n_items))
        self.bx = dict()
        self.c_gap = 100
        self.lb = 0
        self.ub = 1

    def construct_master_base(self):
        """Create the Gurobi MILP model used for the master problem"""
        model = gp.Model('knapsack_master')
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', int(self.time_limit / 10))
        zs = model.addVars(self.n_samples, self.n_items, vtype=GRB.BINARY)
        db = model.addVars(self.n_items, self.n_features, vtype=GRB.CONTINUOUS, lb=self.b_min, ub=self.b_max)
        b_pos = model.addVars(self.n_items, self.n_features, vtype=GRB.CONTINUOUS, lb=0.0)
        b_neg = model.addVars(self.n_items, self.n_features, vtype=GRB.CONTINUOUS, lb=0.0)
        theta = model.addVars(self.n_samples, vtype=GRB.CONTINUOUS, ub=1, lb=0)
        tot_lost = gp.quicksum(theta[i] for i in range(self.n_samples)) / self.n_samples
        lasso_reg = gp.quicksum(b_pos[j, l] + b_neg[j, l] for j in range(self.n_items) for l in range(self.n_features))
        model.setObjective(tot_lost - self.beta * lasso_reg, GRB.MAXIMIZE)
        for i in range(self.n_samples):
            # Create capacity constraints
            model.addConstr(gp.quicksum(zs[i, j] * self.weights[j] for j in range(self.n_items)) <= self.capacity)
            # Create loss constraint zs
            model.addConstr(theta[i] <= gp.quicksum(zs[i, j] * self.costs[i, j] / self.opt_cost[i] for j in range(self.n_items)))

        # Lasso linearization
        model.addConstrs((b_pos[j, l] - b_neg[j, l] - db[j, l] == self.b_fix[j, l] for j in range(self.n_items) for l in range(self.n_features)))

        # Warm-start initialization
        for j in range(self.n_items):
            for l in range(self.n_features):
                db[j, l].Start = 0.0

        for i in range(self.n_samples):
            for j in range(self.n_items):
                zs[i, j].Start = self.init_sol[i, j]

        return model, zs, db, theta

    def set_bx(self, db):
        """Define the expression bx using b variables and the features values"""
        for i in range(self.n_samples):
            for j in range(self.n_items):
                self.bx[(i, j)] = gp.quicksum((db[j, l] + self.b_fix[j, l]) * self.features[i, l] for l in range(self.n_features))

        return 0

    def add_ep_constraints(self, model, zs, theta):
        """Add constraints (14b)-(14c) to the master problem"""
        lbd = model.addVars(self.zk.keys(), vtype=GRB.BINARY)
        # Lower level constraints
        for (i, k) in self.zk.keys():
            model.addConstr(theta[i] <= gp.quicksum(self.zk[(i, k)][j] * self.costs[i, j] / self.opt_cost[i] for j in range(self.n_items)) + lbd[(i, k)])
            model.addConstr(gp.quicksum((self.bx[(i, j)] / self.opt_cost[i]) * (zs[i, j] - self.zk[(i, k)][j]) for j in range(self.n_items)) >= - self.bigM * (1 - lbd[i, k]) + self.eps)

        return model

    def get_db_solution(self, db):
        """Get the current predictor from db current solution"""
        for j in range(self.n_items):
            for l in range(self.n_features):
                self.cb_sol[j, l] = db[j, l].X + self.b_fix[j, l]

        return 0

    def get_zs_solution(self, zs):
        """Get the current zs variables values"""
        for i in range(self.n_samples):
            for j in range(self.n_items):
                self.mp_sol[i, j] = zs[i, j].X

        return 0

    def solve_mp(self):
        """Solve the master problem"""
        model, zs, db, theta = self.construct_master_base()
        self.set_bx(db)
        model = self.add_ep_constraints(model, zs, theta)
        model.optimize()
        self.get_db_solution(db)
        self.get_zs_solution(zs)

        return model.objVal, db, zs

    def update_bounds(self, cub, clb):
        """Update current lower and upper bounds"""
        self.ub = cub * (self.ub > cub) + self.ub * (self.ub < cub)
        if clb > self.lb:
            self.lb = clb
            self.b_sol = self.cb_sol

        return 0

    def update_zk(self, zk_solutions):
        """Update the dictionary containing the current visited extreme points"""
        update = False
        for sol in zk_solutions.keys():
            updated, self.zk, self.zk_np = charge_zk(self.n_samples, self.zk_np, self.zk, zk_solutions[sol])
            update = update or updated

        return update

    def execute_ccg(self):
        """Execute the column-and-constraint algorithm"""
        updated = True
        start_time = time.time()
        elapsed_time = 0
        n_iter = 1
        while (self.c_gap >= self.gap_tol) and (n_iter <= self.max_iter) and updated and (elapsed_time < self.time_limit):
            print('### Iteration ' + str(n_iter))
            cub, db, zs = self.solve_mp()
            for i in range(self.n_samples):
                self.predictions[i, :] = get_predictions(self.cb_sol, self.features[i, :])
            sol_sp, clb = get_bilevel_feasible(self.mp_sol, 0, self.costs, self.predictions, self.weights, self.capacity, 1)
            updated = self.update_zk(sol_sp)
            self.update_bounds(cub, clb)
            self.c_gap = 100 * (self.ub - self.lb) / self.ub
            print('UB = ', self.ub)
            print('LB = ', self.lb)
            print('rGAP = ' + str(self.c_gap))
            elapsed_time = time.time() - start_time
            n_iter += 1
            if elapsed_time > self.time_limit:
                print(' ----- Time limit reached ...')
            if not updated:
                print(' ----- Early stopping: No new solutions added ...')

        return self.b_sol


def solve_ieo_knapsack_ccg(features_matrix, costs_matrix, weights_vector, capacity, b_fixed, opt_cost_matrix, opt_sol_matrix,
                           initial_feasible_points, max_number_points, epsilon_pessimistic, bigM_pessimistic, b_min, b_max,
                           lasso_reg_param, n_cuts_b_random, alpha_b_random, time_limit, maximum_iterations, gap_tolerance):
    """
    Create and solve the Gurobi MILP model for the pessimistic IEO 0-1 knapsack problem using column-and-constraint
    generation

    :param features_matrix:         n_samples x n_features features matrix
    :param costs_matrix:            n_samples x n_features features matrix
    :param weights_vector:          vector of knapsack items weights
    :param capacity:                knapsack capacity
    :param b_fixed:                 b value used in the initialization
    :param opt_cost_matrix:         vector of optimal costs for each sample
    :param opt_sol_matrix:          optimal solutions for each sample
    :param initial_feasible_points: dictionary containing initial points of the knapsack polytope for each sample
    :param max_number_points:       --------------
    :param epsilon_pessimistic:     epsilon parameter used in equation (14c)
    :param bigM_pessimistic:        big-M value used in equation (14c)
    :param b_min:                   minimum b value allowed
    :param b_max:                   maximum b value allowed
    :param lasso_reg_param:         lasso regularization weight
    :param n_cuts_b_random:         number of iterations used for the cut generation (section 3.4.4)
    :param alpha_b_random:          parameter used in the cut generation (section 3.4.4)
    :param time_limit:              time limit for the branch-and-cut algorithm
    :param maximum_iterations:      maximum number of iterations used for the column-and-constraint generation algorithm
    :param gap_tolerance:           mip gap tolerance

    :return: solution for b (predictor)
    """
    (n_samples, n_items) = np.shape(costs_matrix)
    zk_initial = generate_opt_points(initial_feasible_points, n_samples, n_items, weights_vector, capacity, costs_matrix)

    exp_instance = KnapsackCCG(features_matrix, costs_matrix, opt_cost_matrix, opt_sol_matrix, weights_vector, capacity,
                               b_fixed, epsilon_pessimistic, bigM_pessimistic, b_min, b_max, lasso_reg_param, zk_initial,
                               n_cuts_b_random, alpha_b_random, time_limit, maximum_iterations, gap_tolerance)

    b_sol = exp_instance.execute_ccg()

    return b_sol
