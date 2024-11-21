import numpy as np
from HeterogeneousHiddenMarkovModel import HeterogeneousHiddenMarkovModel


def generate_sequences_heterhmm(n, P_initial_list, distances_list, p1, p2, w0, w1, p3, p4, decay_method='sigmoid'):
    """
    Generate synthetic observations using a Heterogeneous Hidden Markov Model (HMM).
    
    Parameters:
    - n (int): Number of observations to generate.
    - P_initial_list (list of ndarray): List of initial state probability distributions for each observation.
    - distances_list (list of ndarray): List of distances between adjacent sites for each observation.
    - p1, p2, w0, w1, p3, p4 (float): Transition, decay, and emission parameters for the model.
    - decay_method (str): Method for calculating transition probability decay (e.g., 'sigmoid').

    Returns:
    - observations (list of tuples): List of generated observations (y, P_initial, distances)
    """
    model = HeterogeneousHiddenMarkovModel(decay_method=decay_method)
    model.A1 = np.array([[1-p1, p1], [p2, 1-p2]])
    model.A2 = np.array([[p1, -p1], [-p2, p2]])
    model.B  = np.array([[1-p3, p3], [p4, 1-p4]])
    model.w  = [w0, w1]
    
    observations = []
    for i in range(n):
        P_initial = P_initial_list[i]
        distances = distances_list[i]
        y, z = model.predict(P_initial, distances)
        observations.append((y, P_initial, distances))
    return observations


def generate_sequences_bernoulli(scores, n_seq, seed=42):
    '''
    Generate synthetic binary sequences with independent bernoulli trials.
    
    Parameters:
    - scores (array): Marginal probabilities for each site (length seq_len), where scores[i] is the
      probability that site i is in state 1.
    - n_seq (int): Number of sequences to generate.

    Returns:
    - sequences (ndarray): Generated sequences with shape (n_seq, seq_len), where each element
      is 0 or 1, representing the state of each site.
    '''

    random_state = np.random.RandomState(seed=seed) # For reproducibility
    # Initialize output array
    seq_len = scores.shape[0]
    sequences = np.empty((n_seq, seq_len), dtype=int)
    
    # Sample
    for i in range(n_seq):
        sequences[i] = random_state.rand(seq_len) < scores
    
    return sequences


def generate_sequences_bidirectional(scores, distances, n_seq, w0=5, w1=-0.05, n_iter=100, seed=42):
    """
    Generate synthetic binary sequences with bidirectional site-site dependencies.
    
    Parameters:
    - scores (array): Marginal probabilities for each site (length seq_len), where scores[i] is the
      probability that site i is in state 1.
    - distances (array): Distances between adjacent sites (length seq_len), with first element set to be 0.
    - n_seq (int): Number of sequences to generate.
    - w0, w1 (float): Base interaction strength and interaction strength decay rate according to distance.
    - n_iter (int): Number of Gibbs sampling iterations for each sequence.
    - seed (int): Seed for reproducibility.
    
    Returns:
    - sequences (ndarray): Generated sequences with shape (n_seq, seq_len), where each element
      is 0 or 1, representing the state of each site.
    """

    random_state = np.random.RandomState(seed=seed) # For reproducibility
    
    seq_len = scores.shape[0]
    scores  = np.clip(scores, 1e-16, 1 - 1e-16)     # Clamp scores for numerical stability

    # Compute external fields and interaction strengths
    h = np.log(scores / (1 - scores))               # External fields
    J = 1/(1+np.exp(-(w0 + w1*distances)))          # Decaying interaction strengths
    
    # Initialize output array
    sequences = np.empty((n_seq, seq_len), dtype=int)
    
    for i in range(n_seq):
        # Initialize sequence randomly
        s = random_state.choice([-1, 1], size=seq_len)
        
        # Gibbs sampling
        for _ in range(n_iter):
            for j in range(seq_len):
                total_field = h[j]
                if j > 0:
                    total_field += J[j] * s[j-1]
                if j < seq_len-1:
                    total_field += J[j+1] * s[j+1]
                prob = 1 / (1 + np.exp(-total_field))
                s[j] = 1 if random_state.rand() < prob else -1
        
        # Convert {-1, +1} to {0, 1}
        sequences[i] = (s.copy() + 1) // 2 
    
    return sequences


def generate_synthetic_data(sample_size, max_seq_len, method='bidirectional', seed=42):
    """
    Generate synthetic feature and target sequences.
    
    Parameters:
    - sample_size (int): Number of sequences to generate.
    - max_seq_len (int): Maximum sequence length for each sample.
    - method (str): Sequence generation method, either 'bidirectional' or 'bernoulli'.
    - seed (int): Seed for reproducibility.

    Returns:
    - features (list of ndarray): List of feature arrays, each with shape (seq_len, 3),
      containing scores, distances, and positions.
    - targets (list of ndarray): List of binary target sequences, each with length seq_len.
    """

    random_state = np.random.RandomState(seed=seed)  # For reproducibility
    
    targets, features = [], []
    for _ in range(sample_size):
        # Generate random sequence length, score (methylation potential), and distance
        seq_len = random_state.randint(5, max_seq_len)
        scores  = np.array([random_state.dirichlet(alpha=[0.5, 0.5])[0] for _ in range(seq_len)])
        distances = np.concatenate(([0], random_state.randint(1, 200, seq_len - 1)))
        positions = np.arange(seq_len)

        # Combine scores, distances into a feature array
        feature_seq = np.column_stack((scores, distances, positions))  # Shape: (seq_len, 3)

        # Generate the target sequence based on the specified method
        if method == 'bidirectional':
            target_seq = generate_sequences_bidirectional(scores, distances, 1)[0]
        elif method == 'bernoulli':
            target_seq = generate_sequences_bernoulli(scores, 1)[0]
        else:
            raise ValueError("Invalid method. Choose either 'bidirectional' or 'bernoulli'.")

        features.append(feature_seq)
        targets.append(target_seq)

    return features, targets

