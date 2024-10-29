import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm # For progress bar

class HeterogeneousMarkovChain:
    def __init__(self, n_states:int=2, decay_method:str="sigmoid", sigmoid_alpha:float=1.0):
        """
        Initialize the Heterogeneous Markov Chain model.

        Parameters:
        - n_states: Number of states for each time point in the Markov chain.
        - decay_method: Decay method to use ('sigmoid' or 'trunc_exp'), both should be monotonic increasing
        - sigmoid_alpha: Steepness adjustment factor for sigmoid. Values >1 make the function steeper.
        """
        
        self.n_states = n_states
        self.decay_method  = decay_method
        self.sigmoid_alpha = sigmoid_alpha
        self._set_decay_function()
        
        # the dtype setting is important, to avoid casting (assign float to int array round it to int..)
        self.estimate_ = None                                       # Optimal estimate from the optimization
        self.estimate_list = None                                   # List of estimates for n_starts, every element is a tuple of (result, init_guess)
        self.optim_est_idx = None                                   # Index of the optimal estimate in the list
        self.loss_history  = None                                   # List of log-likelihood histories for n_starts, every element is a list of log-likelihoods for each iteration
        self.params_history= None                                   # List of parameter histories for n_starts, every element is a list of parameter values for each iteration

        self.A1 = np.zeros((n_states, n_states),dtype=np.float64)   # Placeholder for transition probabilities (distance-independent part)
        self.A2 = np.zeros((n_states, n_states),dtype=np.float64)   # Placeholder for transition probabilities (distance-dependent part)
        self.w  = np.array([0.0, 1.0], dtype=np.float64)             # Placeholder for decay parameters (intercept and slope)

    def _set_decay_function(self):
        """Set decay function based on decay method."""
        if self.decay_method == 'sigmoid':
            self._decay_function = self._sigmoid
        elif self.decay_method == 'trunc_exp':
            self._decay_function = self._trunc_exp
        else:
            raise ValueError("Invalid decay method specified.")
    
    def _sigmoid(self, x):
        """
        Sigmoid function (Adjusted sigmoid when sigmoid_alpha is not 1,0), handles overflow.
        > For numpy double, the range is (-1.79769313486e+308, 1.79769313486e+308)
        To avoid overflow, truncate x within (-709, 709) so that `np.exp(x)` is within that range
        Link: https://stackoverflow.com/questions/47044541/how-to-avoid-an-overflow-in-numpy-exp
        """
        x = np.clip(self.sigmoid_alpha*x, -709, 709)

        return 1.0/(1 + np.exp(-x))
    
    def _trunc_exp(self, x):
        """Truncated exponential function to avoid overflow."""
        x = np.clip(x, -709, 709)

        return np.clip(np.exp(x), -np.inf, 1)
    
    def _get_decay_factors(self, distances):
        """
        Generate transition tensor based on distance sequence.

        Parameters:
        - distances: distance sequence, shape (T,) with first element as 0 (empty)

        Returns:
        - decay_factors: decay coefficients before A2, shape (T,) with first element as 0 (empty)
        """
        
        lambda_ = self.w[1] * distances + self.w[0]
        decay_factors    = self._decay_function(lambda_)
        decay_factors[0] = 0        # Ensure the first element remains 0 if necessary

        return decay_factors

    def init_params(self, init_prob=None, init_w=None):
        """
        Initialize the model parameters.

        Parameters:
        - init_prob: Initial state probability, shape (n_states,), use random if None
        - init_w: Initial decay parameters, shape (2,), use random if None
        """

        init_prob = np.random.random(size=self.n_states) if init_prob is None else init_prob
        init_w    = np.random.normal(size=2) if init_w is None else init_w
        init_w[1] = -np.abs(init_w[1]) # prior knowledge: increase dist -> less dependency, w_1 is negative
        
        return np.concatenate([init_prob, init_w])

    def fit(self, observations, init_prob=None, init_w=None, n_starts:int=1, verbose:bool=False, verbose_result:bool=False):
        """
        Fit the model using given observations.
        to make the model more robust, we can run the optimization multiple times and choose the best result.
        for reproducibility check, please set np.random.seed(seed) outside and specify init_w/prob explicitly

        Parameters:
        - observations: List of tuples, each containing
            - y: State sequence, shape (T,)
            - P_initial: Initial state probability, shape (n_states,)
            - distances: Distance sequence, shape (T,) with first element as 0 (empty)
        - init_prob: Initial state probability for optimization, shape (n_states,), use random if None
        - init_w: Initial decay parameters for optimization, shape (2,), use random if None
        - n_starts: Number of random starts for optimization
        - verbose: Print progress if True
        - verbose_result: Print optimization result if True
        """

        def objective(trans_param):
            self._update_trans_model(trans_param)
            loss = self._trans_neglog_lik(observations)
            if verbose: # not a local variable, access from outer scope
                loss_list.append(loss)
                params_list.append(trans_param)
            return loss
        
        # Initialize variable to store the optimal loss and parameters
        estimate_list = []      # List of estimates for n_starts, every element is a tuple of (result, init_guess)
        loss_history  = []      # List of log-likelihood histories for n_starts, every element is a list of log-likelihoods for each iteration
        params_history = []      # List of parameter histories for n_starts, every element is a list of parameter values for each iteration
        optim_loss = np.inf     # Optimal loss found
        optim_idx  = -1         # Index of the optimal estimate in the list
        
        for i in tqdm(range(n_starts), disable=not verbose):
            loss_list = [] # clean before collect loss
            params_list= [] # clean before collect parameters
            init_guess= self.init_params(init_prob, init_w)
            result = minimize(objective, init_guess,
                              method='L-BFGS-B', bounds=((0, 1), (0, 1), (None, None), (None, 0)))
            estimate_list.append([result, init_guess])
            
            if optim_loss > result.fun:
                optim_loss = result.fun
                optim_idx  = i
                
            if verbose:
                print(f"converge: {result.success}; current loss:{result.fun}; minimum loss: {optim_loss}; estimates: {np.round(result.x, 4)}")
                loss_history.append(loss_list)
                params_history.append(params_list)
            if verbose_result:
                print(f"random init: {init_guess}")
                print(result)

        # Ensure model parameters are updated to the optimal ones found
        if optim_idx >= 0:
            self.estimate_list = estimate_list
            self.optim_est_idx = optim_idx
            self.loss_history  = loss_history
            self.params_history = params_history
            
            optim_est, _ = estimate_list[optim_idx]
            self.estimate_ = optim_est
            self._update_trans_model(optim_est.x)

    def _update_trans_model(self, trans_param):
        self.A1[:] = [[1 - trans_param[0], trans_param[0]], [trans_param[1], 1 - trans_param[1]]]
        self.A2[:] = [[trans_param[0], -trans_param[0]], [-trans_param[1], trans_param[1]]]
        #print(f"Before update (id, value): {id(self.w)}, {self.w}")
        #print("Intended update values:", trans_param[2:4])
        self.w[:]  = trans_param[2:4]
        #print(f"After update (id, value): {id(self.w)}, {self.w}")

    def _trans_neglog_lik(self, observations):
        """
        Calculate the log likelihood of a series of observations.

        Parameters:
        - observations: List of tuples, each containing
            - y: State sequence, shape (T,)
            - P_initial: Initial state probability, shape (n_states,)
            - distances: Distance sequence, shape (T,) with first element as 0 (empty)

        Returns:
        - total_loglik: Total log likelihood of the observations
        """

        total_loglik = 0
        for y, P_initial, distances in observations:
            loglik = np.log(P_initial[y[0]])
            decay_factors = self._get_decay_factors(distances)
            _transition_tensor = self.A1 + self.A2 * decay_factors[:, None, None]

            for t in range(1, len(y)):
                p_transition = _transition_tensor[t, y[t-1], y[t]]
                loglik += np.log(max(p_transition, 5e-324)) # Avoid log(0)
            total_loglik += loglik

        return -total_loglik
    
    def predict(self, P_initial, distances, pred_seed=None, verbose=False):
        """
        Predict a state sequence for given initial state probability and distances.

        Parameters:
        - P_initial: initial state probabilities, shape (n_states,)
        - distances: distance sequence, shape (T,) with first element as 0 (empty)
        - pred_seed: random seed for prediction
        - verbose: print progress if True

        Returns:
        - y: Predicted state sequence, shape (T,)
        """
        self.pred_rng = np.random.RandomState(seed=pred_seed)       # self.pred_rng.get_state()[1][0] to get the specified seed

        T = len(distances)
        y = np.zeros(T, dtype=int)

        decay_factors = self._get_decay_factors(distances)
        _transition_tensor = self.A1 + self.A2 * decay_factors[:, None, None]

        y[0] = self.pred_rng.choice(self.n_states, p=P_initial)  # Choose initial state
        if verbose:
            print(f"Initial state: {y[0]}; init_prob: {P_initial};")

        for t in range(1, T):
            A = _transition_tensor[t, :, :]
            y[t] = self.pred_rng.choice(self.n_states, p=A[y[t-1]])
            if verbose: # site 0 is the site with initial probability
                print(f't={t}/{T-1}: state {y[t]}; trans_prob: {A}; distance: {d};')
            
        return y

