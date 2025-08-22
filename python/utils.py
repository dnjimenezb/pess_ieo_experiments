""" Module with different useful function used along the present implementation
"""
import numpy as np
import pandas as pd
import os


def check_folder(output_path, new_folder):
    """Check if folder exists, if not create it. Return the new folder path"""
    new_folder_path = os.path.join(output_path, new_folder)
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
        print('New directory created: ', new_folder_path)

    return new_folder_path

def check_folder_path(list_folders):
    """Check if complete path exists, if not, create all the required folders"""
    prev = ''
    for folder in list_folders:
        prev = check_folder(prev, folder)

    return prev

def save_b(b, suffix='', output_folder='', instance_name=''):
    """Save b matrices in the indicated output folder, for the indicated instance"""
    (n_items, n_features) = b.shape
    cols = [str(x) for x in range(n_features)]
    b_df = pd.DataFrame(b, columns=cols)
    filename = os.path.join(output_folder, instance_name + suffix + '.csv')
    b_df.to_csv(filename, index=False)

    return 0

def get_predictions(b_matrix, features):
    """Compute predicted knapsack costs given a b matrix and a features matrix"""
    (n_items, n_features) = b_matrix.shape
    predictions = np.zeros(n_items)
    for j in range(n_items):
        predictions[j] = sum(b_matrix[j, l] * features[l] for l in range(n_features))

    return predictions

def get_percentage_loss(objective_value):
    """Compute the percentage loss given an objective value"""
    return 100 * (1 - objective_value)

def charge_zk(n_samples, zk_np, zk, zk_new):
    """Check whether the visited point is already included in the zk dict.
       If yes, include it. Otherwise, omit it"""
    update_flag = True
    if np.sum(zk_np) == 0:
        for i in range(n_samples):
            zk[(i, 0)] = zk_new[i, :]
            zk_np[i] += 1
        return update_flag
    else:
        sk = 0
        for i in range(n_samples):
            flag = True
            NK_i = int(zk_np[i])
            for k in range(NK_i):
                if np.array_equal(zk[(i, k)], zk_new[i, :]):
                    flag = False
                    sk += 1
                    break

            if flag:
                zk[(i, NK_i)] = zk_new[i, :]
                zk_np[i] += 1

        if sk == n_samples:
            update_flag = False

    return update_flag, zk, zk_np