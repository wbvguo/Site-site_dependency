import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import binned_statistic, pearsonr
from sklearn.metrics import mean_absolute_error


def sort_seq_mat(seq_mat, rows, col_index=0):
    """
    Recursively sorts rows of a matrix using a divide-and-conquer strategy to ensure that
    rows with the first non-NA values appear at the top.

    Parameters:
    - seq_mat (numpy.ndarray): A 2D array of binary values (n_seqs, n_sites)
    - rows (list of int): Indices of rows to be sorted.
    - col_index (int): The current column index used for sorting (default is 0).

    Returns: Sorted row indices based on the first non-NA appearance.
    """
    # Base case: Stop recursion when only one row is left or all columns have been checked
    if len(rows) <= 1 or col_index >= seq_mat.shape[1]:
        return rows

    # Divide rows into non-NA and NA groups based on the current column
    non_na_group = [row for row in rows if not np.isnan(seq_mat[row, col_index])]
    na_group = [row for row in rows if np.isnan(seq_mat[row, col_index])]

    # Recursively sort both groups using the next column
    sorted_non_na = sort_seq_mat(seq_mat, non_na_group, col_index + 1)
    sorted_na = sort_seq_mat(seq_mat, na_group, col_index + 1)

    # Conquer step: Combine the sorted groups
    return sorted_non_na + sorted_na


def compute_mean_state(seq_mat):
    """
    Computes the mean state of each site across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of mean state values for each site, ignoring NaNs. (n_sites, )
    """
    count = np.nansum(seq_mat, axis=0)
    total_count= np.sum(~np.isnan(seq_mat), axis=0)
    mean_state = count/total_count
    return mean_state


def count_adjacent_pairs(seq_mat):
    """
    count the occurrences of each pair type for each pair of adjacent columns in a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: A 2D array of counts for each pair type (4, n_sites - 1)
    """
    adjacent_pairs = seq_mat[:, :-1] * 2 + seq_mat[:, 1:]

    # Count occurrences of each pair type for each pair of adjacent columns
    counts = np.array([
        np.sum(adjacent_pairs == 0, axis=0),  # 00
        np.sum(adjacent_pairs == 1, axis=0),  # 01
        np.sum(adjacent_pairs == 2, axis=0),  # 10
        np.sum(adjacent_pairs == 3, axis=0)   # 11
    ])

    return counts


def compute_state_similarity(seq_mat):
    """
    Computes the probability of adjacent sites having the same state across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of probabilities for adjacent sites having the same state (n_sites - 1, )
    """
    num_sites = seq_mat.shape[1]
    same_state_probs = []
    for i in range(num_sites - 1):
        same_state_count = np.sum(seq_mat[:, i] == seq_mat[:, i + 1])
        prob_same_state = same_state_count / seq_mat.shape[0]
        same_state_probs.append(prob_same_state)
    return np.array(same_state_probs)


def compute_site_cor(seq_mat):
    """
    Computes the correlation coefficient between adjacent sites across a set of sequences.
    
    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of correlation coefficients for adjacent sites (n_sites - 1, )
    """
    num_sites = seq_mat.shape[1]
    correlations = []
    for i in range(num_sites - 1):
        site_i = seq_mat[:, i]
        site_j = seq_mat[:, i + 1]
        corr = np.corrcoef(site_i, site_j)[0, 1]
        correlations.append(corr)
    return np.array(correlations)


def compute_adjacent_entropy(seq_mat):
    """
    Computes the joint entropy of adjacent sites across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of joint entropy values for each pair of adjacent sites (n_sites - 1, )
    """
    num_sites = seq_mat.shape[1]
    entropies = []
    for i in range(num_sites - 1):
        site_i = seq_mat[:, i]
        site_j = seq_mat[:, i + 1]
        counts = {'00': np.sum((site_i == 0) & (site_j == 0)),
                  '01': np.sum((site_i == 0) & (site_j == 1)),
                  '10': np.sum((site_i == 1) & (site_j == 0)),
                  '11': np.sum((site_i == 1) & (site_j == 1))}
        total_pairs = sum(counts.values())
        probs = {state: count / total_pairs for state, count in counts.items()}
        entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        entropies.append(entropy)
    return np.array(entropies)


def compute_mutual_info(seq_mat):
    """
    Computes the mutual information between adjacent sites across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of mutual information values for each pair of adjacent sites (n_sites - 1, )
    """
    num_sites = seq_mat.shape[1]
    mutual_infos = []
    for i in range(num_sites - 1):
        site_i = seq_mat[:, i]
        site_j = seq_mat[:, i + 1]
        counts = {'00': np.sum((site_i == 0) & (site_j == 0)),
                  '01': np.sum((site_i == 0) & (site_j == 1)),
                  '10': np.sum((site_i == 1) & (site_j == 0)),
                  '11': np.sum((site_i == 1) & (site_j == 1))}
        total_pairs = sum(counts.values())
        joint_probs = {state: count / total_pairs for state, count in counts.items()}
        # joint entropy
        joint_entropy = -sum(p * np.log2(p) for p in joint_probs.values() if p > 0)
        # individual entropy
        p_i, p_j = np.mean(site_i == 1), np.mean(site_j == 1)
        entropy_i = -((1 - p_i) * np.log2(1 - p_i) + p_i * np.log2(p_i)) if 0 < p_i < 1 else 0
        entropy_j = -((1 - p_j) * np.log2(1 - p_j) + p_j * np.log2(p_j)) if 0 < p_j < 1 else 0

        # Mutual information: I(X; Y) = H(X) + H(Y) - H(X, Y)
        mutual_info = entropy_i + entropy_j - joint_entropy
        mutual_infos.append(mutual_info)

    return np.array(mutual_infos)


def bootstrap_correlation(x, y, n_bootstrap=1000, confidence=95, seed=None):
    """
    Calculate the correlation, bootstrapped standard deviation, and confidence interval.
    
    Parameters:
    - x (array-like): First variable.
    - y (array-like): Second variable.
    - n_bootstrap (int): Number of bootstrap samples. Default is 1000.
    - confidence (float): Confidence level for the interval. Default is 95.
    
    Returns: dict: Dictionary with correlation, std, lower CI, and upper CI.
    """
    # Calculate correlation
    np.random.seed(seed)
    corr, _ = pearsonr(x, y)
    
    # Bootstrap correlations
    boot_corrs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(x), size=len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]
        boot_corr, _ = pearsonr(x_sample, y_sample)
        boot_corrs.append(boot_corr)
    
    # Calculate standard deviation and confidence interval
    boot_std = np.std(boot_corrs)
    lower_ci = np.percentile(boot_corrs, (100 - confidence) / 2)
    upper_ci = np.percentile(boot_corrs, 100 - (100 - confidence) / 2)
    
    return {
        "correlation": corr,
        "boot_std": boot_std,
        f"CI_lower": lower_ci,
        f"CI_upper": upper_ci
    }


def compute_binned_similarity(targets, features, lengths, bin_size=10, bin_edge=None):
    """
    Computes binned mean and standard deviation of site similarity probabilities based on distance.

    Parameters:
    - features (list of numpy.ndarray): A list where each element contains the feature matrix for a sequence. 
                                         The second column contains the distances between adjacent sites.
    - targets (list of numpy.ndarray): A list where each element contains the binary target states for a sequence.
    - lengths (list of int): A list of sequence lengths for each corresponding sequence in `features` and `targets`.
    - bin_size (int): The size of the distance bins.
    - bin_edge (list): A list of bin edges to use for grouping distances (default is None).

    Returns: tuple: (bin_means, bin_stds, bin_edges)
    - bin_means (numpy.ndarray): Mean similarity probabilities in each bin.
    - bin_stds (numpy.ndarray): Standard deviation of similarity probabilities in each bin.
    - bin_edges (numpy.ndarray): The edges of the bins used for grouping distances.
    """
    all_distances, all_similarities = [], []
    for feature_seq, target_seq, seq_len in zip(features, targets, lengths):
        all_distances.extend(feature_seq[1:seq_len, 1])  # Extract distances
        all_similarities.extend((target_seq[:seq_len-1] == target_seq[1:seq_len]).astype(int))  # Similarities

    # Bin distances and calculate statistics
    if bin_edge is not None:
        bin_edges = np.array(bin_edge)
    else:
        bin_edges = np.arange(np.min(all_distances) - bin_size, np.max(all_distances) + bin_size, bin_size)
    bin_means, _, _ = binned_statistic(all_distances, all_similarities, statistic='mean', bins=bin_edges)
    bin_stds, _, _ = binned_statistic(all_distances, all_similarities, statistic='std', bins=bin_edges)
    
    return bin_means, bin_stds, bin_edges


def plot_binned_similarity(datasets, title=None, xlab="Distance between Sites", ylab="Probability of Same State",
                           color = 'steelblue', plot_size=(12, 6), dpi=300, save_path=None):
    """
    Generates scatter plots with linear fit for binned mean comparisons, with shaded confidence intervals.

    Parameters:
    - datasets (list): List of tuples (bin_means, bin_stds, bin_edges) for each subplot.
    - title (str): Title for the entire figure.
    - xlab (str): Label for the x-axis.
    - ylab (str): Label for the y-axis.
    - plot_size (tuple): Size of the plot (default: (12, 6)).
    - dpi (int): DPI for the plot (default: 300).
    - save_path (str): Path to save the plot (default: None).
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=plot_size, dpi=dpi, sharey=True)
    axes = [axes] if len(datasets) == 1 else axes
    title_list = title if isinstance(title, list) and len(title) == len(datasets) else [""] * len(datasets)
    color_list = color if isinstance(color, list) and len(color) == len(datasets) else [color] * len(datasets)
    xlab_list  = xlab if isinstance(xlab, list) and len(xlab) == len(datasets) else [xlab] * len(datasets)
    ylab_list  = ylab if isinstance(ylab, list) and len(ylab) == len(datasets) else [ylab] * len(datasets)
    
    for ix, (bin_means, bin_stds, bin_edges) in enumerate(datasets):
        ax = axes[ix]

        x = bin_edges[:-1] + np.diff(bin_edges) / 2  # Midpoints of bin edges
        y = bin_means
        y_err = bin_stds

        # Calculate statistics for annotation
        correlation, p_value = pearsonr(x[~np.isnan(y)], y[~np.isnan(y)])
        slope, intercept= np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)
        regression_line = slope * x + intercept
        p_value_str     = f"{p_value:.2e}" if p_value < 0.01 else f"{p_value:.2f}"
        intercept_str   = f"{intercept:.4f}" if intercept < 0 else f"+ {intercept:.4f}"
        annotation_str  = (f"$y = {slope:.4f}x {intercept_str}$\n"
                           f"$r = {correlation:.2f}$, $p$ = {p_value_str}")

        ax.set_box_aspect(1)
        # Scatter plot with shaded confidence interval and regression line
        ax.plot(x, y, 'o', color=color_list[ix], label="Binned mean")
        ax.plot(x, y, color=color_list[ix], linewidth=1.5, linestyle="-")  # Line connecting bin means
        ax.fill_between(x, y - y_err, y + y_err, color=color_list[ix], alpha=0.2, label="Mean ± SD")
        ax.plot(x, regression_line, color="red", linewidth=2, label="Linear fit")

        # Plot customization
        ax.set_xlabel(xlab_list[ix], weight="medium")
        ax.set_title(title_list[ix], weight="medium")
        ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)

        # Annotation for correlation and regression details
        ax.annotate(annotation_str, 
                    xy=(0.98, 0.98), 
                    xycoords="axes fraction", 
                    color="black",
                    verticalalignment="top", 
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Shared y-axis label
    axes[0].set_ylabel(ylab_list[ix], weight="medium")

    # Final layout adjustments
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_scatter_comparison(datasets, main_title=None, xlab="x", ylab="y", 
                            color="steelblue", plot_size=(12, 6), dpi=300, save_path=None):
    """
    Generate scatter plots matrix with linear fit for comparison.

    Parameters:
    - datasets (list): List of tuples (x_data, y_data, sub title) for each subplot.
    - xlab (str): Label for the x-axis
    - ylab (str): Label for the y-axis
    - main_title (str): Title for the entire figure.
    - plot_size (tuple): Size of the plot (default: (12, 6)).
    - dpi (int): DPI for the plot (default: 300).
    - save_path (str): Path to save the plot (default: None).
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=plot_size, dpi=dpi, sharey=True)
    fig.suptitle(main_title, fontsize=18, weight="bold")
    
    axes = [axes] if len(datasets) == 1 else axes
    color_list = color if isinstance(color, list) and len(color) == len(datasets) else [color] * len(datasets)
    xlab_list  = xlab if isinstance(xlab, list) and len(xlab) == len(datasets) else [xlab] * len(datasets)
    ylab_list  = ylab if isinstance(ylab, list) and len(ylab) == len(datasets) else [ylab] * len(datasets)

    for ix, (x, y, title) in enumerate(datasets):
        ax = axes[ix]

        # Calculate statistics for annotation
        correlation, p_value = pearsonr(x, y)
        slope, intercept= np.polyfit(x, y, 1)
        regression_line = slope * x + intercept
        p_value_str     = f"{p_value:.2e}" if p_value < 0.01 else f"{p_value:.2f}"
        intercept_str   = f"{intercept:.4f}" if intercept < 0 else f"+ {intercept:.4f}"
        annotation_str  = (f"$y = {slope:.4f}x {intercept_str}$\n"
                           f"$r = {correlation:.2f}$, $p$ = {p_value_str}")
        
        ax.set_box_aspect(1)
        # Scatter plot and regression line
        ax.scatter(x, y, alpha=0.7, edgecolor="k", color=color_list[ix], s=40)
        ax.plot(x, regression_line, color="red", linewidth=2, label="Linear fit")

        # Plot customization
        ax.set_xlabel(xlab_list[ix], weight="medium")
        ax.set_title(title, weight="medium")
        ax.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.7)

        # Annotation for correlation and regression details
        ax.annotate(annotation_str, 
                    xy=(0.98, 0.98), 
                    xycoords="axes fraction", 
                    color="black",
                    verticalalignment="top",  
                    horizontalalignment="right", 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Shared y-axis label
    axes[0].set_ylabel(ylab_list[ix], weight="medium")

    # Final layout adjustments
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_density_scatter_with_marginals(predicted_averages, true_value, title=None, ylab="WGBS methylation", xlab="Prediction Average",
                                        plot_size=(10, 10), save_path=None, marginal_fill='blue', left_marginal=False):
    """
    Generate a density scatter plot with a top marginal KDE plot and color bar to the right 
    for predicted averages vs. true methylation values, including the regression equation.

    Parameters:
    - predicted_averages (array-like): Predicted averages for each site.
    - true_value (array-like): True methylation values.
    - title (str, optional): Title for the plot.
    - save_path (str, optional): Path to save the plot.
    """
    # Calculate regression line, correlation, p-value, and MAE
    slope, intercept= np.polyfit(predicted_averages, true_value, 1)
    regression_line = slope * predicted_averages + intercept
    correlation, p_value = pearsonr(predicted_averages, true_value)
    mae = mean_absolute_error(predicted_averages, true_value)

    # Format regression equation
    intercept_str = f"{intercept:.4f}" if intercept < 0 else f"+ {intercept:.4f}"
    regression_eq = f"y = {slope:.4f}x {intercept_str}"

    # Set up the figure and grid specification
    width_ratios = [0.2, 1, 1, 1, 1, 0.25] if left_marginal else [1, 1, 1, 1, 1, 0.25]
    fig  = plt.figure(figsize=plot_size)
    grid = plt.GridSpec(6, 6, hspace=0.3, wspace=0.4,  # Adjust hspace to control overall vertical spacing
                        width_ratios=width_ratios,
                        height_ratios=[0.5, 1, 1, 1, 1, 0.2])  # Reduce top plot height for better alignment

    # Add a title to the entire figure
    if title:
        fig.suptitle(title, fontsize=18, y=0.95)

    # Main density scatter plot with KDE
    main_ax = fig.add_subplot(grid[1:5, 0:5], aspect='equal')
    kde = sns.kdeplot(x=predicted_averages, y=true_value, ax=main_ax, cmap="viridis", fill=True, thresh=0)

    # Plot the regression line
    main_ax.plot(predicted_averages, regression_line, color="red", linestyle="solid", linewidth=2)

    # Annotations
    annotation_text = (f"{regression_eq}\n"
                       f"Correlation: {correlation:.2f}\n"
                       f"p-value: {p_value:.2e}\n"
                       f"MAE: {mae:.3f}")
    main_ax.text(0.02, 0.98, annotation_text, transform=main_ax.transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Set limits and labels
    main_ax.set_xlim(0, 1)
    main_ax.set_ylim(0, 1)
    main_ax.set_xlabel(xlab, fontsize=16)
    main_ax.set_ylabel(ylab, fontsize=16)
    # main_ax.legend(loc='upper left', fontsize=14)
    main_ax.tick_params(axis='both', labelsize=12)

    # Top marginal KDE plot
    top_ax = fig.add_subplot(grid[0, 0:5], sharex=main_ax)
    sns.kdeplot(predicted_averages, ax=top_ax, color=marginal_fill, fill=True)
    top_ax.set_ylabel('Density', fontsize=14)
    top_ax.tick_params(axis="x", labelbottom=False, labelsize=12)
    top_ax.tick_params(axis="y", labelbottom=False, labelsize=12)
    top_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    top_ax.set_xlim(0, 1)
    # Adjust position to reduce spacing between top marginal and 2D plot
    main_pos = main_ax.get_position()
    top_ax.set_position([main_pos.x0, main_pos.y0 + main_pos.height * 1.03, main_pos.width, main_pos.height * 0.2])
    

    # Optional left marginal KDE plot
    if left_marginal:
        left_ax = fig.add_subplot(grid[1:5, 0], sharey=main_ax)
        sns.kdeplot(y=true_value, ax=left_ax, color='#4682B4', fill=True)
        left_ax.invert_xaxis()  # Flip the left marginal plot
        left_ax.set_xlabel('Density', labelpad=0, fontsize=14)  # Place the label at the bottom with padding
        left_ax.set_ylabel(f'Marginal distribution of {ylab}', fontsize=18)
        left_ax.tick_params(axis="x", labeltop=False, labelbottom=True, labelsize=12)  # Flip ticks to the top
        left_ax.tick_params(axis='y', which='both', left=False, labelleft=False, labelsize=12)
        left_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        left_ax.set_ylim(0, 1)
        left_ax.set_position([main_pos.x0 - main_pos.width * 0.35, main_pos.y0, main_pos.width * 0.2, main_pos.height])

    # Color bar to the right of the scatter plot
    cbar_ax = fig.add_subplot(grid[1:5, 5])
    cbar = fig.colorbar(kde.collections[0], cax=cbar_ax, orientation="vertical", aspect=40)
    cbar.set_label('', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format colorbar labels with two digits
    cbar_ax.set_position([main_pos.x1 + 0.01, main_pos.y0, main_pos.height * 0.01, main_pos.height])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def count_pair_states_seq(targets, features, pos_arr=None, id_col=['distance']):
    """
    Count the occurrences of each pair type for each pair of adjacent columns in a set of sequences.
    - targets (list of numpy.ndarray): A list of binary target states
    - features (list of numpy.ndarray): A list of feature matrix
    - pos_arr (list of numpy.ndarray): A list of position array
    """
    if pos_arr:
        output = [
            [obs[1][i+1, 1], obs[0][i], obs[0][i+1], obs[1][i, 0], obs[1][i+1, 0], obs[2][i], obs[2][i+1]]
            for obs in zip(targets, features, pos_arr)
            for i in range(len(obs[0]) - 1)
        ]

        df = (pd.DataFrame(output, columns=['distance', 'state_l', 'state_r', 'meth_l', 'meth_r', 'pos_l', 'pos_r'])
              .astype({'distance' : int, 'state_l': int, 'state_r': int, 'meth_l': float, 'meth_r': float, 'pos_l': int, 'pos_r': int}))
    else:
        output = [
            [obs[1][i+1, 1], obs[0][i], obs[0][i+1], obs[1][i, 0], obs[1][i+1, 0]]
            for obs in zip(targets, features)
            for i in range(len(obs[0]) - 1)
        ]
        df = (pd.DataFrame(output, columns=['distance', 'state_l', 'state_r', 'meth_l', 'meth_r'])
              .astype({'distance': int, 'state_l': int, 'state_r': int, 'meth_l': float, 'meth_r': float}))

    df_count  = df.groupby(id_col).size().reset_index(name='count')
    df_unique = (df.groupby(id_col + ['state_l', 'state_r']).size().reset_index(name='count')
                 .sort_values(by=id_col + ['state_l', 'state_r']).reset_index(drop=True))
    
    return df, df_count, df_unique


def filter_df_sites(df_count, df_unique, depth_threshold=100, id_col = ['distance']):
    values_with_high_depth = df_count.loc[df_count['count'] >= depth_threshold, 'distance'].tolist()
    
    required_pairs = {(0, 0), (0, 1), (1, 0), (1, 1)}
    values_with_all_pairs = (
        df_unique.
        groupby(id_col)
        .apply(lambda group: required_pairs.issubset(set(zip(group['state_l'], group['state_r']))))
        .loc[lambda x: x]
        .index.tolist()
    )
    
    df_filter = (
        df_unique[(df_unique['distance'].isin(values_with_all_pairs)) & 
                  (df_unique['distance'].isin(values_with_high_depth))].
        sort_values(by='distance').
        reset_index(drop=True)
    )
    group_totals = df_filter.groupby(['distance'])['count'].transform('sum')
    df_filter['frequency'] = df_filter['count'] / group_totals
    
    return df_filter


def compute_df_sameprob(group):
    return group.loc[group['state_l'] == group['state_r'], 'frequency'].sum()


def compute_df_entropy(group):
    probabilities = group['frequency'] / group['frequency'].sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def compute_df_mutinfo(group):
    total_count = group['count'].sum()        # Total counts in the group
    joint_prob = group['count'] / total_count # Joint probability P(X, Y)
    # Marginal probabilities P(X) and P(Y)
    p_x = group.groupby('state_l')['count'].sum() / total_count
    p_y = group.groupby('state_r')['count'].sum() / total_count

    # Map marginal probabilities to rows
    group = group.copy()
    group['p_x'] = group['state_l'].map(p_x)
    group['p_y'] = group['state_r'].map(p_y)

    # Calculate mutual information: P(x, y) * log(P(x, y) / (P(x) * P(y)))
    mi = np.sum(joint_prob * np.log(joint_prob / (group['p_x'] * group['p_y'])))
    return mi


def compute_df_sitecor(group):
    # Repeat each site pair based on the 'count' to represent data distribution
    repeated_data = group.loc[group.index.repeat(group['count'])]
    # Calculate correlation between state_l and state_r
    correlation = np.corrcoef(repeated_data['state_l'], repeated_data['state_r'])[0, 1]
    return correlation


def compute_df_dep(df_filter, id_col = ['distance']):
    df_entropy = (
        df_filter
        .groupby(id_col)
        .apply(compute_df_entropy, include_groups=False)
        .reset_index(name='entropy')
    )
    df_mutinfo = (
        df_filter
        .groupby(id_col)
        .apply(compute_df_mutinfo, include_groups=False)
        .reset_index(name='mutinfo')
    )
    df_sitecor = (
        df_filter
        .groupby(id_col)
        .apply(compute_df_sitecor, include_groups=False)
        .reset_index(name='sitecor')
    )
    df_sameprob = (
        df_filter
        .groupby(id_col)
        .apply(compute_df_sameprob, include_groups=False)
        .reset_index(name='sameprob')
    )

    from functools import reduce
    df_dep = reduce(lambda left, right: pd.merge(left, right, on=id_col, how='outer'), 
                    [df_entropy, df_mutinfo, df_sitecor, df_sameprob])
    return df_dep

