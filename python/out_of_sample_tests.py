"""Module containing the functions used to execute the out-of-sample experiments
"""
from data_reading import *
from utils import *
import argparse


def get_method_os(b_method, costs, features, weights, capacity):
    """
    Compute the percentual out-of-sample loss for the selected method

    :param b_method:   b matrix of the tested predictor
    :param costs:      ns_test x n_items out-of-sample costs matrix
    :param features:   ns_test x n_features out-of-sample features matrix
    :param weights:    vector of knapsack items weights
    :param capacity:   knapsack capacity

    :return:           ns_test vector of the percentual out-of-sample losses
    """
    (ns_test, n_items) = costs.shape
    (_, n_features) = features.shape
    results_array = np.zeros(ns_test)
    for i in range(ns_test):
        predictions = get_predictions(b_method, features[i, :])
        _, obj_val = get_bilevel_feasible_single(np.zeros((1, n_items)), 0, costs[i, :], predictions, weights, capacity, 1)
        results_array[i] = get_percentage_loss(obj_val)

    return results_array

def do_out_of_sample_tests(n_items, ns_train, ns_test):
    """
    Execute the out-of-sample experiments for all the tested methods.

    :param n_items:   Number of items of the 0-1 knapsack instances (only 10 and 20 are allowed)
    :param ns_train:  Number of training samples used
    :param ns_test:   Number of test samples used to perform the out-of-sample experiments

    :return: Create .csv files containing the out-of-sample losses for each method
    """
    methods = ['bLR', 'bSPOp', 'bSPOpCH', 'bIEO']
    instances_folder = os.path.join("data", "ni_%i"%n_items, "train", "ns_%i"%ns_train)
    b_matrices_folder = os.path.join("output", "b_matrices", "ni_%i"%n_items, "ns_%i"%ns_train)
    output_folder_list = ["output", "out_of_sample_results", "ni_%i"%n_items, "ns_%i"%ns_train]
    save_path = check_folder_path(output_folder_list)
    train_instances = os.listdir(instances_folder)
    for instance in train_instances:
        print('###########################################')
        print('... Out-of-sample experiments for instance:', instance)
        print('###########################################')
        path_data = os.path.join(instances_folder, instance)
        _, n_features, b_gt, deg, eps, weights, capacity = read_data(path_data, ns_train, mode='test')
        instance_name = instance.split('.')[0]
        features_test, costs_test = gen_test_samples(ns_test, n_items, n_features, b_gt, deg, eps)
        df_results = pd.DataFrame()
        for method in methods:
            print('.... Solving for method:', method)
            try:
                b_method_path = os.path.join(b_matrices_folder, instance_name + '_' + method + '.csv')
                b_method_matrix = pd.read_csv(b_method_path).values
                df_results[method] = get_method_os(b_method_matrix, costs_test, features_test, weights, capacity)
            except:
                print('Method matrix (' + method + ') not found for instance ' + instance_name + ', skipping...')

        df_path = os.path.join(save_path, instance_name + '_out_of_sample.csv')
        df_results.to_csv(df_path, index=True)

    return 0

def construct_final_report(n_items, ns_train_array):
    """
    Construct the final report of different metrics for the out-of-sample experiments

    :param n_items:         Number of items of the 0-1 knapsack instances (only 10 and 20 are allowed)
    :param ns_train_array:  Lists of the number of training samples used to compute b matrices

    :return: Create a .csv file containing the final report of different metrics for the out-of-sample experiments
    """
    methods = ['bLR', 'bSPOp', 'bSPOpCH', 'bIEO']
    indicators = ['Mean in-sample loss', 'Mean out-of-sample loss', 'StdDev out-of-sample loss']
    cols = ['N. Instance', 'N. Samples', 'N. Items', 'N. Features', 'Cap/Tot_weight', 'Delta', 'Eps']
    for ind in indicators:
        for method in methods:
            cols.append(ind + ' (' + method + ')')

    df_final_report = pd.DataFrame(columns=cols)
    k = 0
    for ns_train in ns_train_array:
        instances_folder = os.path.join("data", "ni_%i" % n_items, "train", "ns_%i" % ns_train)
        b_matrices_folder = os.path.join("output", "b_matrices", "ni_%i" % n_items, "ns_%i" % ns_train)
        out_of_sample_folder = os.path.join("output", "out_of_sample_results", "ni_%i" % n_items, "ns_%i" % ns_train)
        train_instances = os.listdir(instances_folder)
        for instance in train_instances:
            path_data = os.path.join(instances_folder, instance)
            _, n_features, features, costs, weights, capacity = read_data(path_data, ns_train, mode='train')
            current_row = dict()
            elements = instance.split('_')
            instance_name = instance.split('.')[0]
            path_out_of_sample = os.path.join(out_of_sample_folder, instance_name + '_out_of_sample.csv')
            df_out_of_sample = pd.read_csv(path_out_of_sample)
            current_row['N. Samples'] = ns_train
            current_row['Cap/Tot_weight'] = str(capacity) + '/' + str(np.sum(weights))
            current_row['N. Instance'] = elements[1].replace('inst', '')
            current_row['N. Items'] = n_items
            current_row['N. Features'] = n_features
            current_row['Delta'] = elements[4].replace('delta', '')
            current_row['Eps'] = elements[5].replace('eps', '')
            for method in methods:
                try:
                    b_method_path = os.path.join(b_matrices_folder, instance_name + '_' + method + '.csv')
                    b_method = pd.read_csv(b_method_path).values
                    method_objval = get_objective_value(b_method, ns_train, weights, capacity, costs, features)
                    current_row['Mean in-sample loss' + ' (' + method + ')'] = get_percentage_loss(method_objval)
                    current_row['Mean out-of-sample loss' + ' (' + method + ')'] = np.mean(df_out_of_sample[method].values)
                    current_row['StdDev out-of-sample loss' + ' (' + method + ')'] = np.std(df_out_of_sample[method].values)
                except:
                    print('Out of samples results for method :', method, ' not found for instance: ', instance, ', skipping...')

            df_final_report.loc[k] = pd.Series(current_row)
            k += 1

    path_final_report_list = ["output", "final_stats", "ni_%i" % n_items]
    path_final_report = check_folder_path(path_final_report_list)
    df_final_report.to_csv(os.path.join(path_final_report,"final_stats.csv"), index=False)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=10, help="Number of knapsack items (10 or 20)")
    parser.add_argument("--ns_train", nargs="+", type=int, default=[100, 200, 300, 400, 500],
                        help="Training samples (100, 200, 300, 400 or 500)")
    parser.add_argument("--ns_test", type=int, default=1000, help="Number of test samples")
    args = parser.parse_args()
    n_items = args.n_items
    ns_test = args.ns_test
    ns_train_array = args.ns_train
    for ns_train in ns_train_array:
        do_out_of_sample_tests(
            n_items, ns_train, ns_test
        )
    construct_final_report(
        n_items, ns_train_array
    )
