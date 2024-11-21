import numpy as np
from scipy.stats import binned_statistic


def compute_mean_state(sequences):
    """
    Computes the mean state of each site across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of mean state values for each site, ignoring NaNs. (n_sites, )
    """
    count = np.nansum(sequences, axis=0)
    total_count= np.sum(~np.isnan(sequences), axis=0)
    mean_state = count/total_count
    return mean_state


def compute_state_similarity(sequences):
    """
    Computes the probability of adjacent sites having the same state across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of probabilities for adjacent sites having the same state (n_sites - 1, )
    """
    num_sites = sequences.shape[1]
    same_state_probs = []
    for i in range(num_sites - 1):
        same_state_count = np.sum(sequences[:, i] == sequences[:, i + 1])
        prob_same_state = same_state_count / sequences.shape[0]
        same_state_probs.append(prob_same_state)
    return np.array(same_state_probs)


def compute_site_cor(sequences):
    """
    Computes the correlation coefficient between adjacent sites across a set of sequences.
    
    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of correlation coefficients for adjacent sites (n_sites - 1, )
    """
    num_sites = sequences.shape[1]
    correlations = []
    for i in range(num_sites - 1):
        site_i = sequences[:, i]
        site_j = sequences[:, i + 1]
        corr = np.corrcoef(site_i, site_j)[0, 1]
        correlations.append(corr)
    return np.array(correlations)


def compute_adjacent_entropy(sequences):
    """
    Computes the joint entropy of adjacent sites across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of joint entropy values for each pair of adjacent sites (n_sites - 1, )
    """
    num_sites = sequences.shape[1]
    entropies = []
    for i in range(num_sites - 1):
        site_i = sequences[:, i]
        site_j = sequences[:, i + 1]
        counts = {'00': np.sum((site_i == 0) & (site_j == 0)),
                  '01': np.sum((site_i == 0) & (site_j == 1)),
                  '10': np.sum((site_i == 1) & (site_j == 0)),
                  '11': np.sum((site_i == 1) & (site_j == 1))}
        total_pairs = sum(counts.values())
        probs = {state: count / total_pairs for state, count in counts.items()}
        entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        entropies.append(entropy)
    return np.array(entropies)


def compute_mutual_info(sequences):
    """
    Computes the mutual information between adjacent sites across a set of sequences.

    Parameters: A 2D array of binary values (n_seqs, n_sites)
    Returns: An array of mutual information values for each pair of adjacent sites (n_sites - 1, )
    """
    num_sites = sequences.shape[1]
    mutual_infos = []
    for i in range(num_sites - 1):
        site_i = sequences[:, i]
        site_j = sequences[:, i + 1]
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


def compute_binned_similarity(features, targets, lengths, bin_size=10):
    """
    Computes binned mean and standard deviation of site similarity probabilities based on distance.

    Parameters:
    - features (list of numpy.ndarray): A list where each element contains the feature matrix for a sequence. 
                                         The second column contains the distances between adjacent sites.
    - targets (list of numpy.ndarray): A list where each element contains the binary target states for a sequence.
    - lengths (list of int): A list of sequence lengths for each corresponding sequence in `features` and `targets`.
    - bin_size (int): The size of the distance bins.

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
    bin_edges = np.arange(np.min(all_distances) - bin_size, np.max(all_distances) + bin_size, bin_size)
    bin_means, _, _ = binned_statistic(all_distances, all_similarities, statistic='mean', bins=bin_edges)
    bin_stds, _, _ = binned_statistic(all_distances, all_similarities, statistic='std', bins=bin_edges)
    
    return bin_means, bin_stds, bin_edges

