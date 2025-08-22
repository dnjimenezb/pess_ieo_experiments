""" Script that generates the plots present in the manuscript.
IEO matrices computation and out-of-sample testing should be executed first.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import os
import argparse


def generate_boxplot(n_items, indicator):
    """
    Generates the bloxplot charts present in the manuscript.

    :param n_items:     number of knapsack items
    :param indicator:   indicator to be plotted (Choose between 'Mean in-sample loss', 'Mean out-of-sample loss',
                        'StdDev out-of-sample loss')

    :return: Creates a .pdf file with the boxplot chart present in the manuscript.
    """
    path_report = os.path.join("output", "final_stats", "ni_%i"%n_items)
    df_report = pd.read_csv(os.path.join(path_report, "final_stats.csv"))
    deg_list = [1, 3, 5]
    labelsize = 20
    space = 0.15
    if n_items == 10:
        legend = ['LR', 'SPO+', 'SPO+(CH)', 'IEO']
        methods = ['LR', 'SPOp', 'SPOp(CH)', 'IEO']
        position = [-1.5, -0.5, 0.5, 1.5]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    elif n_items == 20:
        legend = ['LR', 'SPO+', 'IEO']
        methods = ['LR', 'SPOp', 'IEO']
        position = [-1.1, 0, 1.1]
        colors = ['tab:blue', 'tab:orange', 'tab:purple']
    else:
        print('Please for n_items select either 10 or 20 ....')
        return 0

    ind_max_value = 0
    for method in methods:
        col_name = indicator + ' (b' + method + ')'
        ind_max_value = max(ind_max_value, df_report[col_name].max())

    fig, ax1 = plt.subplots(1, 3, figsize=(12, 4.5))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.135, top=0.9, wspace=0.03, hspace=0.4)
    box_param = dict(whis=(5, 95), widths=0.15, patch_artist=True,
                     flierprops=dict(marker='.', markeredgecolor='black',
                                     fillstyle=None), medianprops=dict(color='black'))

    k = 0
    for delta in deg_list:
        data = df_report[df_report['Delta'] == delta]
        nb_groups = data['N. Samples'].nunique()
        df_pivot = data.pivot(columns=['N. Samples'])
        ax1[k].set_title('$\delta = $' + str(delta), fontsize=20)
        ind_dict = dict()
        bp_dict = dict()
        kk = 0
        for method in methods:
            try:
                col_name = indicator + ' (b' + method + ')'
                ind_dict['b'+method] = [df_pivot[col_name][var].dropna() for var in df_pivot[col_name]]
                bp_dict[method] = ax1[k].boxplot(ind_dict['b'+method], positions=np.arange(nb_groups) + position[kk] * space,
                                 boxprops=dict(facecolor=colors[kk]), showfliers=False, **box_param)
                kk += 1
            except KeyError:
                print('Results for ' + method + ' were not found.')

        ax1[k].set_xticks(np.arange(nb_groups))
        ax1[k].set_xticklabels([f'{label}' for label in data['N. Samples'].unique()])
        ax1[k].grid(visible=True, which='major', axis='y', ls='--')
        ax1[k].tick_params(axis='x', labelsize=labelsize)
        ax1[k].set_ylim(0, ind_max_value)
        ticks_space = int(10 * ind_max_value / 6) / 10
        ele_ticks = np.arange(0, ind_max_value, ticks_space)
        if k == 0:
            ax1[k].set_yticks(ele_ticks)
            yticks_fmt = dict(axis='y', which='both', labelsize=labelsize)
            ax1[k].tick_params(**yticks_fmt)
        else:
            ax1[k].set_yticks(ele_ticks)
            ax1[k].tick_params(axis='y', which='both', left=False, right=False, bottom=False,
                top=False, labelbottom=False, labelleft=False, labelright=False)

        # Format axes labels
        ylabel_fmt = dict(size=18, labelpad=20)
        xlabel_fmt = dict(size=18, labelpad=3)
        ax1[k].set_xlabel('$n$', **xlabel_fmt)
        if k == 0:
            ax1[k].set_ylabel('$\\bar{\ell}_{IEO}$', **ylabel_fmt)

        k += 1

    legend_body = [bp_dict[x]["boxes"][0] for x in bp_dict.keys()]
    plt.legend(legend_body, legend, loc='upper right', fontsize=13)
    path_plot = os.path.join(path_report, "boxplots.pdf")
    plt.savefig(path_plot, dpi=180, facecolor='w', edgecolor='w', orientation='portrait', format='pdf',
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
    plt.show()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=10, help="Number of knapsack items (10 or 20)")
    parser.add_argument("--ind", type=str, default='Mean out-of-sample loss', help="Measure to plot")
    args = parser.parse_args()
    n_items = args.n_items
    indicator = args.ind
    generate_boxplot(n_items, indicator)
