import numpy as np

class HomogeneousHiddenMarkovModel:
    def __init__(self, n_states:int=2, n_emits:int=2, max_iter:int=1000, conv_target:str="params", tolerance:float=1e-5, init_seed:int=None):
        """
        Initialize the Homogeneous Hidden Markov Model with discrete emissions.

        Parameters:
        - n_states: Number of states
        - n_emits:  Number of possible emissions
        - max_iter: maximum number of iterations
        - tolerance: convergence criterion
        - init_seed: random seed for initialization
        """
        
        self.n_states   = n_states
        self.n_emits    = n_emits
        
        self.max_iter   = max_iter
        self.conv_target= conv_target
        self.tolerance  = tolerance
        self.init_rng   = np.random.RandomState(seed=init_seed)
        
        self.estimate_list = None               # List of estimated parameters from multiple starts
        self.optim_est_idx = None               # Index of the optimal estimate in the list
        self.loglik_history= None               # List of log-likelihood histories for multiple starts
        self.params_history= None               # List of parameter histories for multiple starts
        
        self.A = np.zeros((n_states, n_states),dtype=np.float64)  # Placeholder for Transition probabilities
        self.B = np.zeros((n_states, n_emits), dtype=np.float64)  # Placeholder for Emission probabilities

    def init_params(self, init_A=None, init_B=None):
        """
        Initializes the transition and emission probabilities with provided or random values.

        Parameters:
        - init_A: Initial transition matrix, shape (n_states, n_states) with rows summing to 1
        - init_B: Initial emission matrix, shape (n_states, n_emits) with rows summing to 1
        """

        self.A = self.init_rng.rand(self.n_states, self.n_states) if init_A is None else init_A
        self.B = self.init_rng.rand(self.n_states, self.n_emits) if init_B is None else init_B
        
        self.A /= (self.A.sum(axis=1, keepdims=True) + np.finfo(float).eps)
        self.B /= (self.B.sum(axis=1, keepdims=True) + np.finfo(float).eps)

    def forward(self, y, P_initial):
        """
        Computes the forward probabilities.

        Parameters:
        - y: observed sequence, shape (T,) with elements in {0, 1, ..., n_emits-1}
        - P_initial: initial state probabilities, shape (n_states,) with sum equal to 1

        Returns:
        - alpha: the forward probabilities, shape (T, n_states)
        """
        
        T = len(y)
        alpha = np.zeros((T, self.n_states))
        alpha[0, :] = P_initial * self.B[:, y[0]] # element-wise product [pi[i] * b[i,y[0]] for i in range(n_states)]
        for t in range(1, T):
            alpha[t, :] = (alpha[t-1, :] @ self.A) * self.B[:, y[t]]
            # for j in range(self.n_states):
            #     alpha[t, j] = np.dot(alpha[t-1, :], self.A[:, j]) * self.B[j, y[t]]
        return alpha
    
    def backward(self, y):
        """
        Computes the backward probabilities.

        Parameters:
        - y: observed sequence, shape (T,) with elements in {0, 1, ..., n_emits-1}

        Returns:
        - beta: the backward probabilities, shape (T, n_states)
        """
        
        T = len(y)
        beta = np.zeros((T, self.n_states))
        beta[T-1, :] = 1  # Initialization

        for t in range(T-2, -1, -1):
            beta[t, :] = (self.A @ (self.B[:, y[t+1]] * beta[t+1, :]))
            # for j in range(self.n_states):
            #     beta[t, j] = np.sum(self.A[j, :] * self.B[:, y[t+1]] * beta[t+1, :])
        return beta

    def fit(self, observations, init_A=None, init_B=None, n_starts:int=1, max_iter:int=None, conv_target:str="params", tolerance:float=None, verbose:bool=False):
        """
        Fits the model parameters to a list of observation sequences.

        Parameters:
        - observations: list of observation sequences, each a tuple (y, P_initial)
        - init_A: Initial transition matrix, shape (n_states, n_states) with rows summing to 1
        - init_B: Initial emission matrix, shape (n_states, n_emits) with rows summing to 1
        - n_starts: number of random starts
        - max_iter: maximum number of iterations for each start, if None, use the default value
        - tolerance: convergence criterion, if None, use the default value
        - verbose: whether to print progress
        """

        estimate_list = []      # List of estimated results for n_starts
        loglik_history= []      # List of log-likelihood histories for n_starts, every element is a list of log-likelihoods for each iteration
        params_history= []      # List of parameter histories for n_starts, every element is a list of parameters for each iteration
        optim_loglik  = -np.inf # The largest log-likelihood among n_starts
        optim_idx     = -1      # Index of the estimate with the largest log-likelihood among n_starts
        
        for j in range(n_starts):
            if n_starts > 1:
                print(f"Random starts: {j}")
            self.init_params(init_A=init_A, init_B=init_B)
            n_iters, loglik, loglik_list, params_list = self.run_em(observations, max_iter=max_iter, tolerance=tolerance, verbose=verbose)
            is_converge = n_iters < self.max_iter-1
            if verbose:
                if is_converge:
                    print(f"Converged successfully in {n_iters + 1} iterations; log-likelihood: {loglik}")
                else:
                    print(f"Did not converge after {n_iters + 1} iterations; log-likelihood: {loglik}")    
                loglik_history.append(loglik_list)
                params_history.append(params_list)
            
            estimate_list.append([n_iters, loglik, self._get_param()])
            if optim_loglik < loglik:
                optim_loglik = loglik
                optim_idx = j
        
        # Ensure model parameters are updated to the optimal ones found
        if optim_idx >= 0:
            self.estimate_list = estimate_list
            self.optim_est_idx = optim_idx
            if verbose:
                self.loglik_history = loglik_history
                self.params_history = params_history
            
            n_iters, loglik, optim_param = estimate_list[optim_idx]
            self._update_model(optim_param)
    
    def run_em(self, observations, max_iter=None, conv_target=None, tolerance=None, verbose=False):
        """
        Runs the EM algorithm for a given set of observations.

        Parameters:
        - observations: list of observation sequences, each a tuple (y, P_initial)
        - max_iter: maximum number of iterations, if None, use the default value
        - tolerance: convergence criterion, if None, use the default value
        - verbose: whether to print progress

        Returns:
        - n_iters: number of iterations performed
        - loglik: final log-likelihood
        - loglik_list: list of log-likelihoods for each iteration
        """
        
        self.max_iter   = self.max_iter if max_iter is None else max_iter
        self.conv_target= self.conv_target if conv_target is None else conv_target
        self.tolerance  = self.tolerance if tolerance is None else tolerance

        old_loglik  = 0
        old_params  = [np.inf]*len(self._get_param())
        
        loglik_list= []
        params_list= []
        for i in range(self.max_iter):
            # E-step: Expectation given current parameters
            params  = self._get_param()
            loglik, A_num, A_den, B_num, B_den = self.e_step(observations)

            if verbose:
                print(
                    f'{self._color("#iterations", color="red")}: {i}; '
                    f'{self._color("log-likelihood", color="blue")}: {loglik:.6f}; '
                    f'{self._color("estimates", color="green")}: [{", ".join([f"{p:.4f}" for p in params])}]'
                )
                loglik_list.append(loglik)
                params_list.append(params)

            # Check for convergence from the last step
            conv_score = np.abs(params - old_params).max() if self.conv_target == "params" else np.abs(loglik - old_loglik)
            if conv_score < self.tolerance:
                break

            # M-step: Maximization
            self.m_step(A_num, A_den, B_num, B_den)
            old_loglik = loglik
            old_params = params
        
        return i, loglik, loglik_list, params_list

    def e_step(self, observations):
        loglik = 0
        A_num, A_den = np.zeros_like(self.A), np.zeros(self.n_states)
        B_num, B_den = np.zeros_like(self.B), np.zeros(self.n_states)
        
        for y, P_initial in observations:
            alpha = self.forward(y, P_initial)
            beta = self.backward(y)
            gamma, xi = self.e_step_1seq(y, alpha, beta)
            
            # the sum of alpha's last element is the likelihood of the sequence
            loglik += np.log(np.sum(alpha[-1])) 
            
            # Accumulate updates for M-step
            A_num += np.sum(xi, axis=0)                 # sum over xi's along the time axis, shape is (n_states, n_states)
            A_den += np.sum(A_num, axis=1)              # sum over A_num's along the time axis, shape is (n_states,)
            for t in range(len(y)):                     # found it's faster than the alternative below
                B_num[:, y[t]] += gamma[t, :]
            B_den += np.sum(gamma, axis=0)
            
            # A_num += np.sum(xi, axis=0)                # sum over xi's along the time axis, shape is (n_states, n_states)
            # A_den += np.sum(gamma[:-1, :], axis=0)     # sum over gamma's along the time axis, shape is (n_states,)
            # #considering gamma[:-1, :]=np.sum(xi, axis=2), reduce computation by summing A_num
            # #np.sum(np.sum(xi, axis=2), axis=0) # T x m x n -> T x m -> m
            # #np.sum(np.sum(xi, axis=0), axis=1) # T x m x n -> m x n -> m

            # for yt in range(self.n_emits):
            #     B_num[:, yt] += np.sum(gamma[y == yt, :], axis=0)
            # B_den += np.sum(gamma, axis=0)
        
        return loglik, A_num, A_den, B_num, B_den

    def e_step_1seq(self, y, alpha, beta):
        """
        Performs the E-step of the EM algorithm.
        
        Parameters:
        - y: array of observed sequence, shape (T,) with elements in {0, 1, ..., n_emits-1}
        - alpha: forward probabilities, shape (T, n_states)
        - beta: backward probabilities, shape (T, n_states)

        Returns:
        - gamma: the state occupation probabilities, shape (T, n_states)
        - xi: the state transition probabilities, shape (T-1, n_states, n_states)
        """

        T = len(y)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        gamma = np.zeros((T, self.n_states))
        
        for t in range(T-1):
            numer = (alpha[t, :, None] * self.A * self.B[:, y[t+1]] * beta[t+1, None, :])
            xi[t] = numer / (numer.sum() + np.finfo(float).eps)
            # denom = np.dot(alpha[t, :], (self.A * self.B[:, y[t+1]].T)) @ beta[t+1, :].T
            # xi[t, :, :] = (alpha[t, :, None] * self.A * self.B[:, y[t+1]].T * beta[t+1, :]) / denom
            # for t in range(T-1):
            #     denom = np.dot(alpha[t, :], self.A) * self.B[:, y[t+1]].T * beta[t+1, :].T
            #     denom = np.sum(denom) + np.finfo(float).eps
            #     for i in range(self.n_states):
            #         numer = alpha[t, i] * self.A[i, :] * self.B[:, y[t+1]].T * beta[t+1, :]
            #         xi[t, i, :] = numer / denom

        
        gamma[:-1, :] = np.sum(xi, axis=2) # use xi to compute the gamma, then compute the last gamma
        gamma[-1, :] = (alpha[-1, :] * beta[-1, :]) / np.dot(alpha[-1, :], beta[-1, :])
        # for t in range(T):
        #     gamma[t, :] = (np.sum(xi[t-1, :, :], axis=0) if t > 0 else
        #                    (alpha[t, :] * beta[t, :]) / 
        #                    (np.dot(alpha[t, :], beta[t, :]) + np.finfo(float).eps))

        return gamma, xi
    
    def m_step(self, A_num, A_den, B_num, B_den):
        """
        Performs the M-step of the EM algorithm.

        Parameters:
        - A_num: numerator of the transition matrix update
        - A_den: denominator of the transition matrix update
        - B_num: numerator of the emission matrix update
        - B_den: denominator of the emission matrix update
        """

        self.A = A_num / (A_den[:, None] + np.finfo(float).eps)
        self.A /= (self.A.sum(axis=1, keepdims=True) + np.finfo(float).eps)
        self.B = B_num / (B_den[:, None] + np.finfo(float).eps)
        self.B /= (self.B.sum(axis=1, keepdims=True) + np.finfo(float).eps)
        
    def predict(self, P_initial, length):
        """
        Generates a sequence of observations based on the model parameters.

        Parameters:
        - P_initial: initial state probabilities, shape (n_states,) with sum equal to 1
        - length: length of the sequence to generate

        Returns:
        - y: the observed sequence, shape (length,)
        - z: the state sequence, shape (length,)
        """
        
        y = np.zeros(length, dtype=int)
        z = np.zeros(length, dtype=int)
        
        # Initial state is chosen based on P_initial, and generate first observation
        z[0] = np.random.choice(self.n_states, p=P_initial)
        y[0] = np.random.choice(self.n_emits, p=self.B[z[0], :])
        
        # Generate the rest of the sequence
        for t in range(1, length):
            # Transition to next state based on A, generate observation based on B
            z[t] = np.random.choice(self.n_states, p=self.A[z[t-1], :])
            y[t] = np.random.choice(self.n_emits, p=self.B[z[t], :])
        
        return y, z
    
    def _get_param(self):
        '''
        Return the model parameters as a 1-D array. (p1, p2, p3, p4)
        '''
        return np.array([self.A[0,1], self.A[1,0], 
                        self.B[0,1], self.B[1,0]])

    def _update_model(self, params):
        '''
        Update the model parameters from a 1-D array. (p1, p2, p3, p4)
        '''
        self.A[:] = [[1-params[0], params[0]], [params[1], 1-params[1]]]
        self.B[:] = [[1-params[2], params[2]], [params[3], 1-params[3]]]
    
    def _color(self, text_str, color="blue"):
        """
        Color for state visualization.
        """
        
        color_dict = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", 
                      "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m", "white": "\033[97m"}

        return color_dict[color] + text_str + "\033[0m"

