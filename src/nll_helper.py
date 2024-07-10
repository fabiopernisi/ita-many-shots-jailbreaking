import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def get_nll(result_dict, all_shots):
    """
    Get the negative log likelihood for the generation of a single attack (out of 50 attacks).

    Returns:
    nlls: list of negative log likelihoods for the shots.
    """
    nlls = []
    for key in all_shots:
        nlls.append(result_dict[key][0][int(key)-1][1])
    return nlls

def bootstrap_confidence_interval_per_point(data, num_bootstrap_samples=1000, confidence_level=0.95):
    """
    Calculate the 95% confidence interval for each point using bootstrap resampling.

    Parameters:
    data: The input data to bootstrap.
    num_bootstrap_samples: Number of bootstrap samples to generate.
    confidence_level: The confidence level for the interval.

    Returns:
    lower_bounds (np.array): Lower bound of the confidence interval for each point.
    upper_bounds (np.array): Upper bound of the confidence interval for each point.
    """
    num_samples, num_points = data.shape
    lower_bounds = np.zeros(num_points)
    upper_bounds = np.zeros(num_points)

    for i in range(num_points):
        bootstrap_samples = np.random.choice(data[:, i], (num_bootstrap_samples, num_samples), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        lower_bounds[i] = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
        upper_bounds[i] = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)

    return lower_bounds, upper_bounds


def plot_nll_results_with_ci_subplot(ax, nll_values, lower_bounds, upper_bounds, title, color, x_values, xlabel=True, ylabel=True):
    x_values = np.array(x_values, dtype=float)
    ax.scatter(x_values, nll_values, label='Average NLL values', color=color)
    ax.plot(x_values, nll_values, color=color)
    ax.fill_between(x_values, lower_bounds, upper_bounds, color=color, alpha=0.2, label='_nolegend_')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    if xlabel:
        ax.set_xlabel('Number of Shots (log scale)', fontsize=18)
    if ylabel:
        ax.set_ylabel('NLL of harmful response', fontsize=18)
    
    ax.set_title(title, fontsize=20)
    
    #ax.legend()
    ax.grid(True, which="both", ls="--")

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.get_minor_formatter().set_scientific(False)
    ax.yaxis.get_minor_formatter().set_useOffset(False)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)


def process_and_plot_nll_dataframes(nll_results_list, titles, num_shot_values_list):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (nll_results_50, num_shot_values) in enumerate(zip(nll_results_list, num_shot_values_list)):
        stacked_nll_results = np.array([get_nll(nll_results_50[j], num_shot_values) for j in range(50)])
        average_nll_results = np.mean(stacked_nll_results, axis=0)
        lower_bounds, upper_bounds = bootstrap_confidence_interval_per_point(stacked_nll_results)
        
        xlabel = True if i >= 3 else False
        ylabel = True if i % 3 == 0 else False
        plot_nll_results_with_ci_subplot(axes[i], average_nll_results, lower_bounds, upper_bounds, titles[i], f'C{i}', num_shot_values, xlabel=xlabel, ylabel=ylabel)
    
    plt.tight_layout()
    plt.show()