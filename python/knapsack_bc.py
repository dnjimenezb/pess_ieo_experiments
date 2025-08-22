"""Module to solve the pessimistic IEO for the 0-1 knapsack using branch-and-cut

This module solves the pessimistic IEO for the 0-1 knapsack using branch-and-cut.
This implementation uses the Gurobi MILP solver and its callback tool to add lazy
constraints in the branch-and-bound tree.
"""

import logging
from single_knapsack import *
from utils import get_predictions


class MPCallback:
    """
        Class managing the addition of cuts in the branch-and-cut algorithm
    """
    def __init__(self, costs, features, weights, capacity, eps, opt_cost, bigM, b_fix, zs, db, theta, lbd, u,
                 zk_initial, alpha, n_try):
        """
        :param costs:       n_samples x n_items costs matrix
        :param features:    n_samples x n_features features matrix
        :param weights:     vector of knapsack items weights
        :param capacity:    knapsack capacity
        :param eps:         epsilon parameter used in equation (14c)
        :param opt_cost:    vector of optimal costs for each sample
        :param bigM:        big-M value used in equation (14c)
        :param b_fix:       b value used in the initialization
        :param zs:          zs variables
        :param db:          db variables (b_fix + db = b)
        :param theta:       theta variables
        :param lbd:         lambda variables
        :param u:           u variables (bz product linearization)
        :param zk_initial:  dictionary containing initial extreme points of the knapsack polytope for each sample
        :param alpha:       parameter used in the cut generation (section 3.4.4)
        :param n_try:       number of iterations used for the cut generation (section 3.4.4)
        """
        self.costs = costs
        self.features = features
        self.weights = weights
        self.capacity = capacity
        self.eps = eps
        self.opt_cost = opt_cost
        self.bigM = bigM
        self.b_fix = b_fix
        self.zs = zs
        self.db = db
        self.theta = theta
        self.lbd = lbd
        self.u = u
        (self.n_samples, self.n_items) = self.costs.shape
        (_, self.n_features) = self.features.shape
        self.predicted_costs = np.zeros((self.n_samples, self.n_items))
        self.zk_dict = zk_initial
        self.alpha = alpha
        self.n_try = n_try
        self.n_init = list(zk_initial.keys())[-1][1] + 1
        self.zk_np = self.n_init * np.ones(self.n_samples)
        self.bx = dict()
        self.bz_prod = dict()
        self.current_best_objval = 0
        self.db_sol = np.zeros((self.n_samples, self.n_items))
        self.db_dict = dict()
        self.zs_dict = dict()

    def __call__(self, model, where):
        """Identify the conditions to add a cut and add it if needed"""
        add_lazy_cut = False
        cut_type = 0
        if where == GRB.Callback.MIPSOL:
            cut_type = 0
            add_lazy_cut = True
        elif (where == GRB.Callback.MIPNODE) and (model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL):
            cut_type = 1
            add_lazy_cut = True
        if add_lazy_cut:
            try:
                self.add_violated_constraints(model, cut_type)
            except Exception:
                logging.exception("Exception occurred in callback")
                model.terminate()

    def add_violated_constraints(self, model, type_c):
        """Read current variables values, compute the cut coefficients, and add them"""
        db_values, zs_values = self.get_db_zs_values(model, type_c)
        self.def_bx_bz()
        n_rep = self.n_try * (type_c == 1) + 1 * (type_c == 0)
        for k in range(n_rep):
            if k == 0:
                db_gen = db_values
            else:
                db_gen = db_values + self.alpha * (np.random.rand(self.n_items, self.n_features) - 0.5)
            self.get_predictions(db_gen)
            current_sol, objective_value = get_bilevel_feasible(
                mp_sol=zs_values,
                eps=self.eps,
                costs=self.costs,
                predictions=self.predicted_costs,
                weights=self.weights,
                capacity=self.capacity,
                type_c=type_c
            )
            self.update_zk(model, current_sol)

            if objective_value > self.current_best_objval:
                #print(" ************* Better feasible solution found: ", objective_value)
                self.db_sol = db_gen
                self.current_best_objval = objective_value
                model.cbLazy(gp.quicksum(self.theta[i] for i in range(self.n_samples)) / self.n_samples >= objective_value)

        return 0

    def update_zk(self, model, sol_dict):
        """Update the dictionary containing the current visited extreme points"""
        update_flag = False
        for name in sol_dict.keys():
            flag = self.charge_zk(model, sol_dict[name])
            update_flag = update_flag or flag

        return not update_flag

    def charge_zk(self, model, zk_new):
        """Check whether the visited point is already included in the zk dict.
        If yes, include it. Otherwise, omit it"""
        update_flag = True
        if np.sum(self.zk_np) == 0:
            for i in range(self.n_samples):
                self.zk_dict[(i, 0)] = zk_new[i, :]
                self.zk_np[i] += 1
            return update_flag
        else:
            sk = 0
            for i in range(self.n_samples):
                flag = True
                NK_i = int(self.zk_np[i])
                for k in range(NK_i):
                    if np.array_equal(self.zk_dict[(i, k)], zk_new[i, :]):
                        flag = False
                        sk += 1
                        break

                if flag:
                    self.zk_dict[(i, NK_i)] = zk_new[i, :]
                    self.zk_np[i] += 1
                    self.add_single_constraints(model, i, NK_i, zk_new[i, :])

            if sk == self.n_samples:
                update_flag = False

        return update_flag

    def add_single_constraints(self, model, i, k, z_new):
        """Add new lazy constraints associated to the new visited point z_new"""
        model.cbLazy(
            self.theta[i] <= gp.quicksum(z_new[j] * self.costs[i, j] / self.opt_cost[i] for j in range(self.n_items))
            + self.lbd[i, k])
        model.cbLazy(
            self.bz_prod[i] / self.opt_cost[i] - gp.quicksum((self.bx[(i, j)] / self.opt_cost[i]) * z_new[j]
                                                             for j in range(self.n_items))
            >= - self.bigM * (1 - self.lbd[i, k]) + self.eps)

        return 0

    def get_db_zs_values(self, model, type_c):
        """Get the current values of db and zs variables"""
        if type_c == 0:
            self.db_dict = model.cbGetSolution(self.db)
            self.zs_dict = model.cbGetSolution(self.zs)
        elif type_c == 1:
            self.db_dict = model.cbGetNodeRel(self.db)
            self.zs_dict = model.cbGetNodeRel(self.zs)
        db_values = np.zeros((self.n_items, self.n_features))
        zs_values = np.zeros((self.n_samples, self.n_items))
        for (j, l) in self.db_dict.keys():
            db_values[j, l] = self.db_dict[(j, l)]
        for (i, j) in self.zs_dict.keys():
            zs_values[i, j] = self.zs_dict[(i, j)]

        return db_values, zs_values

    def get_predictions(self, db_values):
        """Compute the predicted costs using the current db variables values"""
        for i in range(self.n_samples):
            self.predicted_costs[i, :] = get_predictions(self.b_fix + db_values, self.features[i, :])

        return 0

    def def_bx_bz(self):
        """Define the expressions of bz and bx as functions of u and zs variables"""
        for i in range(self.n_samples):
            self.bz_prod[i] = (
                gp.quicksum(
                    self.u[i, j, l] * self.features[i, l] + self.b_fix[j, l] * self.features[i, l] * self.zs[i, j]
                    for j in range(self.n_items) for l in range(self.n_features)
                )
            )
            for j in range(self.n_items):
                self.bx[(i, j)] = (
                        gp.quicksum((self.db[j, l] + self.b_fix[j, l]) * self.features[i, l] for l in range(self.n_features))
                )

        return 0

def solve_ieo_knapsack_bc(features_matrix, costs_matrix, weights_vector, capacity, b_fixed, opt_cost_matrix, opt_sol_matrix,
                          initial_feasible_points, max_number_points, epsilon_pessimistic, bigM_pessimistic, b_min, b_max,
                          lasso_reg_param,n_cuts_b_random, alpha_b_random, time_limit, maximum_iterations, gap_tolerance):
    """
    Create and solve the Gurobi MILP model for the pessimistic IEO 0-1 knapsack problem using branch-and-cut

    :param features_matrix:         n_samples x n_features features matrix
    :param costs_matrix:            n_samples x n_features features matrix
    :param weights_vector:          vector of knapsack items weights
    :param capacity:                knapsack capacity
    :param b_fixed:                 b value used in the initialization
    :param opt_cost_matrix:         vector of optimal costs for each sample
    :param opt_sol_matrix:          optimal solutions for each sample
    :param initial_feasible_points: dictionary containing initial points of the knapsack polytope for each sample
    :param max_number_points:       maximum number of points used for the knapsack polytope
    :param epsilon_pessimistic:     epsilon parameter used in equation (14c)
    :param bigM_pessimistic:        big-M value used in equation (14c)
    :param b_min:                   minimum b value allowed
    :param b_max:                   maximum b value allowed
    :param lasso_reg_param:         lasso regularization weight
    :param n_cuts_b_random:         number of iterations used for the cut generation (section 3.4.4)
    :param alpha_b_random:          parameter used in the cut generation (section 3.4.4)
    :param time_limit:              time limit for the branch-and-cut algorithm
    :param maximum_iterations:      ------
    :param gap_tolerance:           mip gap tolerance

    :return: solution for b (predictor)
    """
    (n_samples, n_items) = np.shape(costs_matrix)
    (_, n_features) = np.shape(features_matrix)
    zk_initial = generate_opt_points(initial_feasible_points, n_samples, n_items, weights_vector, capacity, costs_matrix)
    ik_idx = zk_initial.keys()

    with gp.Env() as env, gp.Model(env=env) as m:
        # Solver configuration
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)
        m.setParam('MIPGap', gap_tolerance)

        # Model variables
        zs = m.addVars(n_samples, n_items, vtype=GRB.BINARY)
        db = m.addVars(n_items, n_features, vtype=GRB.CONTINUOUS)
        b_pos = m.addVars(n_items, n_features, vtype=GRB.CONTINUOUS, lb=0.0)
        b_neg = m.addVars(n_items, n_features, vtype=GRB.CONTINUOUS, lb=0.0)
        u = m.addVars(n_samples, n_items, n_features, vtype=GRB.CONTINUOUS)
        theta = m.addVars(n_samples, vtype=GRB.CONTINUOUS, ub=1, lb=0)
        lbd = m.addVars(n_samples, max_number_points, vtype=GRB.BINARY)
        tot_lost = gp.quicksum(theta[i] for i in range(n_samples)) / n_samples
        lasso_reg = gp.quicksum(b_pos[j, l] + b_neg[j, l] for j in range(n_items) for l in range(n_features))
        bx = dict()
        bz_prod = dict()
        bx_fix = dict()

        #Solver warm-start initialization
        for j in range(n_items):
            for l in range(n_features):
                db[j, l].Start = 0.0

        for i in range(n_samples):
            for j in range(n_items):
                zs[i, j].Start = opt_sol_matrix[i, j]

        for i in range(n_samples):
            # Set bz product
            bz_prod[i] = (
                    gp.quicksum(
                        u[i, j, l] * features_matrix[i, l] + b_fixed[j, l] * features_matrix[i, l] * zs[i, j]
                        for j in range(n_items) for l in range(n_features))
            )
            # Set bx values
            for j in range(n_items):
                bx[(i, j)] = gp.quicksum((db[j, l] + b_fixed[j, l]) * features_matrix[i, l] for l in range(n_features))
                bx_fix[i, j] = sum(b_fixed[j, l] * features_matrix[i, l] for l in range(n_features))

        # Lower level constraints
        for (i, k) in ik_idx:
            m.addConstr(theta[i] <= gp.quicksum(zk_initial[(i, k)][j] * costs_matrix[i, j] / opt_cost_matrix[i] for j in range(n_items)) + lbd[(i, k)])
            m.addConstr(gp.quicksum((bx[(i, j)] / opt_cost_matrix[i]) * (zs[i, j] - zk_initial[(i, k)][j]) for j in range(n_items)) >= - bigM_pessimistic * (1 - lbd[i, k]) + epsilon_pessimistic)

        for i in range(n_samples):
            # Create capacity constraints
            m.addConstr(gp.quicksum(zs[i, j] * weights_vector[j] for j in range(n_items)) <= capacity)
            # Create loss constraint zs
            m.addConstr(theta[i] <= gp.quicksum(zs[i, j] * costs_matrix[i, j] / opt_cost_matrix[i] for j in range(n_items)))

        # Lasso linearization
        m.addConstrs((b_pos[j, l] - b_neg[j, l] - db[j, l] == b_fixed[j, l] for j in range(n_items) for l in range(n_features)))

        # Bz linearization constraints
        m.addConstrs((u[i, j, l] >= b_min * zs[i, j] for i in range(n_samples) for j in range(n_items) for l in range(n_features)))
        m.addConstrs((u[i, j, l] <= b_max * zs[i, j] for i in range(n_samples) for j in range(n_items) for l in range(n_features)))
        m.addConstrs((db[j, l] - u[i, j, l] >= b_min * (1 - zs[i, j]) for i in range(n_samples) for j in range(n_items) for l in range(n_features)))
        m.addConstrs((db[j, l] - u[i, j, l] <= b_max * (1 - zs[i, j]) for i in range(n_samples) for j in range(n_items) for l in range(n_features)))
        m.addConstrs((db[j, l] >= b_min for j in range(n_items) for l in range(n_features)))
        m.addConstrs((db[j, l] <= b_max for j in range(n_items) for l in range(n_features)))

        # Set objective
        m.setObjective(tot_lost - lasso_reg_param * lasso_reg, GRB.MAXIMIZE)

        # Set lazy constraints
        m.Params.LazyConstraints = 1
        cb = MPCallback(costs=costs_matrix, features=features_matrix, weights=weights_vector, capacity=capacity,
                        eps=epsilon_pessimistic, opt_cost=opt_cost_matrix, bigM=bigM_pessimistic, b_fix=b_fixed,
                        zs=zs, db=db, theta=theta, lbd=lbd, u=u, zk_initial=zk_initial, n_try=n_cuts_b_random,
                        alpha=alpha_b_random)
        m.optimize(cb)

        # Get the solutions
        b_sol_1 = np.zeros((n_items, n_features))
        b_sol_2 = cb.db_sol + b_fixed
        for j in range(n_items):
            for l in range(n_features):
                b_sol_1[j, l] = db[j, l].X + b_fixed[j, l]

        obj_val_1 = get_objective_value(b_sol_1, n_samples, weights_vector, capacity, costs_matrix, features_matrix)
        obj_val_2 = get_objective_value(b_sol_2, n_samples, weights_vector, capacity, costs_matrix, features_matrix)

        #Check which of the solutions gets a better loss, if discrepancies are found
        if obj_val_1 > obj_val_2:
            b_sol_final = b_sol_1
        else:
            b_sol_final = b_sol_2

        return b_sol_final
