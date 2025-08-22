"""
Main script to compute pessimistic IEO predictor for the 0-1 knapsack problem
"""
from data_reading import *
from knapsack_bc import solve_ieo_knapsack_bc
from knapsack_ccg import solve_ieo_knapsack_ccg
from utils import *
import logging
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_items", type=int, default=10, help="Number of knapsack items (10 or 20)")
parser.add_argument("--n_samples", nargs="+", type=int, default=[100, 200, 300, 400, 500], help="Training samples (100, 200, 300, 400 or 500)")
parser.add_argument("--method", type=str, default='bc', help="Method to solve pessimistic IEO ('bc' or 'ccg')")
parser.add_argument("--timelimit", type=int, default=1800, help="Time limit in seconds")
args = parser.parse_args()

n_items = args.n_items
ns_array = args.n_samples
method = args.method
timelimit = args.timelimit

exp_params = {
    'initial_feasible_points' : 10,
    'max_number_points' : 2000,     # only for bc
    'epsilon_pessimistic': 1E-5,
    'bigM_pessimistic' : 1,
    'b_min' : -4,
    'b_max' : 4,
    'lasso_reg_param' : 0,
    'n_cuts_b_random' : 1,
    'alpha_b_random' : 0.1,
    'time_limit' : timelimit,
    'maximum_iterations': 100,      # only for ccg
    'gap_tolerance' : 1E-4
}

instance_params = {}

for ns_train in ns_array:
    instances_folder = os.path.join("data", "ni_%i"%n_items, "train", "ns_%i"%ns_train)
    output_folder_list = ["output", "b_matrices", "ni_%i"%n_items, "ns_%i"%ns_train]
    train_instances = os.listdir(instances_folder)

    for instance in train_instances:
        print('###########################################')
        print('... Solving instance :', instance)
        print('###########################################')
        data_path = os.path.join(instances_folder, instance)
        save_path = check_folder_path(output_folder_list)

        _, n_features, features, costs, weights, capacity = read_data(data_path, ns_train, mode='train')
        opt_is_costs, opt_is_sol = calc_opt_in_sample(ns_train, n_items, weights, capacity, costs)
        instance_name = instance.split('.')[0]
        if n_items == 10:
            instance_name_ch = instance.split('_m10_')[0]
            ch_folder = os.path.join("data", "ni_%i"%n_items, "ch_inequalities")
            path_ch = os.path.join(ch_folder, instance_name_ch + '_coefficients.csv')
            b_spop_ch = calc_spop_ch(path_ch, n_items, costs, features, opt_is_sol)
            spopch_objval = get_objective_value(b_spop_ch, ns_train, weights, capacity, costs, features)

        b_lr = calc_b_lr(n_items, n_features, features, costs)
        b_spop = calc_spop(weights, capacity, costs, features, opt_is_sol)
        lr_objval = get_objective_value(b_lr, ns_train, weights, capacity, costs, features)
        spop_objval = get_objective_value(b_spop, ns_train, weights, capacity, costs, features)


        # choosing b fix for warm-start initialization
        if spop_objval > lr_objval:
            b_fix = b_spop
        else:
            b_fix = b_lr

        instance_params['features_matrix'] = features
        instance_params['costs_matrix'] = costs
        instance_params['weights_vector'] = weights
        instance_params['capacity'] = capacity
        instance_params['b_fixed'] = b_fix
        instance_params['opt_cost_matrix'] = opt_is_costs
        instance_params['opt_sol_matrix'] = opt_is_sol

        if method == 'bc':
            b_sol = solve_ieo_knapsack_bc(**instance_params, **exp_params)
        elif method == 'ccg':
            b_sol = solve_ieo_knapsack_ccg(**instance_params, **exp_params)

        try:
            ieo_objval = get_objective_value(b_sol, ns_train, weights, capacity, costs, features)
            save_b(b_sol, '_bIEO', save_path, instance_name)
            save_b(b_lr, '_bLR', save_path, instance_name)
            save_b(b_spop, '_bSPOp', save_path, instance_name)
            if n_items == 10:
                save_b(b_spop_ch, '_bSPOpCH', save_path, instance_name)
            print(' #### In-sample loss :')
            print('LR      :', get_percentage_loss(lr_objval))
            print('SPO+    :', get_percentage_loss(spop_objval))
            if n_items == 10:
                print('SPO+(CH):', get_percentage_loss(spopch_objval))
            print('IEO     :', get_percentage_loss(ieo_objval))

        except Exception:
            logging.exception("Exception occurred: Please select 'bc or 'ccg' as method")
