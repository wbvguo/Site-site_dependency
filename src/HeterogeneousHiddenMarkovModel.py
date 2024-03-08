import numpy as np
from scipy.optimize import minimize

# when p3 p4 approaches to 0.5, it would be hard to find the right values for transition model parameters
# understandable as the emission probabilities are not very informative, adding noise to the sequence

class HeterogeneousHiddenMarkovModel:
    def __init__(self, n_states=2, n_emits=2, decay_method="sigmoid", sigmoid_alpha=1, max_iter=1000, tolerance=1e-5, init_seed=None):
        '''
        initialize the model with the number of states, number of emissions, decay method, sigmoid alpha, max iterations, and tolerance

        Parameters:
        - n_states: Number of states in the Markov chain.
        - n_emits: Number of emissions in the Markov chain.
        - decay_method: Decay method to use ('sigmoid' or 'trunc_exp').
        - sigmoid_alpha: alpha parameter for sigmoid function
        - max_iter: maximum number of iterations for EM algorithm
        - tolerance: tolerance for convergence
        - init_seed: seed for random number generator
        '''
        
        self.n_states = n_states
        self.n_emits  = n_emits
        self.decay_method  = decay_method
        self.sigmoid_alpha = sigmoid_alpha
        self._set_decay_function()
        
        self.max_iter = max_iter
        self.tolerance= tolerance
        self.init_rng = np.random.RandomState(seed=init_seed)       # self.init_rng.get_state()[1][0] to get the specified seed
        
        self.estimate_list = None                                   # List of estimated results for n_starts
        self.optim_est_idx = None                                   # Index of the estimate with the largest log-likelihood among n_starts
        self.loglik_history= None                                   # List of log-likelihood histories for n_starts, every element is a list of log-likelihoods for each iteration
        self.param_history = None                                   # List of parameter histories for n_starts, every element is a list of parameter values for each iteration
        
        self.A1 = np.zeros((n_states, n_states),dtype=np.float64)   # Placeholder for transition probabilities (distance-independent part)
        self.A2 = np.zeros((n_states, n_states),dtype=np.float64)   # Placeholder for transition probabilities (distance-dependent part)
        self.B  = np.zeros((n_states, n_emits), dtype=np.float64)   # Placeholder for emission probabilities
        self.w  = np.array([0.0, 1.0], dtype=np.float64)            # Placeholder for decay parameters (intercept and slope)
        self._transition_tensor = None                              # Placeholder for transition tensor of shape (T, n_states, n_states), first element is 0 matrix
    
    def _set_decay_function(self):
        """Set decay function based on decay method."""
        if self.decay_method == 'sigmoid':
            self._decay_function = self._sigmoid
        elif self.decay_method == 'trunc_exp':
            self._decay_function = self._trunc_exp
        else:
            raise ValueError("Invalid decay method specified.")
    
    def _sigmoid(self, x, return_grad=False):
        """
        Sigmoid function (Adjusted sigmoid when sigmoid_alpha is not 1,0), handles overflow.
        > For numpy double, the range is (-1.79769313486e+308, 1.79769313486e+308)
        To avoid overflow, truncate x within (-709, 709) so that `np.exp(x)` is within that range
        Link: https://stackoverflow.com/questions/47044541/how-to-avoid-an-overflow-in-numpy-exp
        """
        x = np.clip(self.sigmoid_alpha*x, -709, 709)
        fx = 1.0/(1 + np.exp(-x))
        fx_grad = self.sigmoid_alpha * fx * (1 - fx) if return_grad else None

        return fx, fx_grad
    
    def _trunc_exp(self, x, return_grad=False):
        """Truncated exponential function."""
        x = np.clip(x, -709, 709)
        fx = np.clip(np.exp(x), -np.inf, 1)
        fx_grad = np.where(x < 0, fx, 0) if return_grad else None

        return fx, fx_grad
    
    def _get_decay_factors(self, distances, return_grad=False):
        """
        Generate transition tensor based on distance sequence.

        Parameters:
        - distances: distance sequence, shape (T,) with first element as 0 (empty)

        Returns:
        - decay_factors: decay coefficients before A2, shape (T,) with first element as 0 (empty)
        - decay_factors_grad: first-order derivative of decay coefficients (for intercept), shape (T,) with first element as 0 (empty)
        """
        
        lambda_ = self.w[1] * distances + self.w[0]
        decay_factors, decay_factors_grad = self._decay_function(lambda_, return_grad=return_grad)
        decay_factors[0] = 0            # Ensure the first element remains 0 if necessary
        if return_grad:
            decay_factors_grad[0] = 0   # Ensure the first element remains 0 if necessary

        return decay_factors, decay_factors_grad
    
    def init_param(self, init_A1=None, init_B=None, init_w=None, normalize=False):
        """
        Initializes transition probabilities, emission probabilities, and coefficients of distance decay.

        Parameters:
        - init_A1: initial transition probabilities (distance-independent part), A2 can be computed from A1
        - init_B: initial emission probabilities
        - init_w: initial decay coefficients
        - normalize: whether to normalize the transition and emission probabilities to sum to 1
        if None, random values are used
        """
        
        self.A1= self.init_rng.rand(self.n_states, self.n_states) if init_A1 is None else init_A1
        self.B = self.init_rng.rand(self.n_states, self.n_emits)  if init_B  is None else init_B
        self.w = self.init_rng.normal(size=2) if init_w is None else init_w
        
        if normalize:
            self.A1 /= (self.A1.sum(axis=1, keepdims=True) + np.finfo(float).eps)
            self.B  /= (self.B.sum(axis=1,  keepdims=True) + np.finfo(float).eps)
            self.w[1]= -np.abs(self.w[1]) # Ensure slope is negative, so that decay factor decreases as distance increases
        self.A2  = np.eye(self.n_states) - self.A1
    
    def forward(self, y, P_initial):
        """
        Computes the forward probabilities with distance-dependent transition probabilities.
        
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
            A_t = self._transition_tensor[t, :, :]
            alpha[t, :] = (alpha[t-1, :] @ A_t) * self.B[:, y[t]]
            # for j in range(self.n_states):
            #     alpha[t, j] = np.dot(alpha[t-1, :], A_t[:, j]) * self.B[j, y[t]]
        return alpha
    
    def backward(self, y):
        """
        Computes the backward probabilities with distance-dependent transition probabilities.

        Parameters:
        - y: observed sequence, shape (T,) with elements in {0, 1, ..., n_emits-1}

        Returns:
        - beta: the backward probabilities, shape (T, n_states)
        """

        T = len(y)
        beta = np.zeros((T, self.n_states))
        beta[T-1, :] = 1  # Initialization
        
        for t in range(T-2, -1, -1):
            A_tp = self._transition_tensor[t+1, :, :] # A_{t+1}, shape (n_states, n_states)
            beta[t, :] = (A_tp @ (self.B[:, y[t+1]] * beta[t+1, :]))
            # for j in range(self.n_states):
            #     beta[t, j] = np.sum(A_tp[j, :] * self.B[:, y[t+1]] * beta[t+1, :])
        return beta
    
    def fit(self, observations, init_A1=None, init_B=None, init_w=None, n_starts=1, max_iter:int=None, tolerance:float=None, verbose=False, verbose_result=False):
        """
        Fits the model parameters to a list of observation sequences.
        
        Parameters:
        - observations: list of observation sequences, each a tuple (y, P_initial, distances)
            - y: observation sequence of shape (T,)
            - P_initial: initial state probabilities of shape (n_states,)
            - distances: distance sequence of shape (T,) with first element as 0 (empty)
        - init_A1: initial transition probabilities (distance-independent part), if None, use random values
        - init_B: initial emission probabilities, if None, use random values
        - init_w: initial decay coefficients, if None, use random values
        - n_starts: number of random starts for EM algorithm
        - max_iter: maximum number of iterations for EM algorithm, if None, use the default value
        - tolerance: convergence criterion, if None, use the default value
        - verbose: whether to print progress
        - verbose_result: whether to print optimization result of numeric optimization method
        """

        estimate_list = []          # List of estimated results for n_starts
        optim_loglik  = -np.inf     # The largest log-likelihood among n_starts
        optim_idx     = -1          # Index of the estimate with the largest log-likelihood among n_starts
        loglik_history= []          # List of log-likelihood histories for n_starts, every element is a list of log-likelihoods for each iteration
        param_history = []          # List of parameter histories for n_starts, every element is a list of parameter values for each iteration
        
        for j in range(n_starts):
            self.init_param(init_A1=init_A1, init_B=init_B, init_w=init_w)
            n_iters, loglik, loglik_list, param_list = self.run_em(observations, max_iter=max_iter, tolerance=tolerance, verbose=verbose, verbose_result=verbose_result)
            if verbose:
                is_converge = n_iters < self.max_iter-1
                if is_converge:
                    print(f"Converged successfully in {n_iters + 1} iterations; log-likelihood: {loglik}")
                else:
                    print(f"Did not converge after {n_iters + 1} iterations; log-likelihood: {loglik}")
                loglik_history.append(loglik_list)
                param_history.append(param_list)
            
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
                self.param_history  = param_history
            
            n_iters, loglik, optim_param = estimate_list[optim_idx]
            self._update_model(optim_param)
    
    def run_em(self, observations, max_iter=None, tolerance=None, verbose=False, verbose_result=False):
        """
        Runs the EM algorithm for a given set of observations.

        Parameters:
        - observations: list of observation sequences, each a tuple (y, P_initial, distances)
            - y: observation sequence of shape (T,)
            - P_initial: initial state probabilities of shape (n_states,)
            - distances: distance sequence of shape (T,) with first element as 0 (empty)
        - max_iter: maximum number of iterations, if None, use the default value
        - tolerance: convergence criterion, if None, use the default value
        - verbose: whether to print progress
        - verbose_result: whether to print optimization result of numeric optimization method

        Returns:
        - n_iters: number of iterations performed
        - loglik: final log-likelihood
        - loglik_list: list of log-likelihoods for each iteration
        - param_list: list of parameter values for each iteration
        """
        
        self.max_iter = self.max_iter if max_iter is None else max_iter
        self.tolerance= self.tolerance if tolerance is None else tolerance
        
        new_param = self._get_param()
        loglik_list = []
        param_list  = []
        for i in range(self.max_iter):
            old_param = new_param
            loglik  = 0
            B_num   = np.zeros_like(self.B)
            
            # E-step: Expectation
            xi_list = []
            distances_list = []
            for y, P_initial, distances in observations:
                # for each observation sequence, compute the decay factor and transition tensor
                decay_factors, _ = self._get_decay_factors(distances)
                self._transition_tensor = self.A1 + self.A2 * decay_factors[:, None, None]
                alpha = self.forward(y, P_initial)
                beta = self.backward(y)
                gamma, xi = self.e_step(y, alpha, beta)

                loglik += np.log(np.sum(alpha[-1]))
                
                # Accumulate updates for M-step
                for t in range(len(y)):
                    B_num[:, y[t]] += gamma[t, :]
                
                xi_list.append(xi)
                distances_list.append(distances)
            
            if verbose:
                print(f'{self._color("#iterations")}: {i}; {self._color("log-likelihood", color="red")}: {loglik}; estimates: {np.round(old_param, 4)}')
                loglik_list.append(loglik)
                param_list.append(old_param)
            
            # M-step: Maximization
            self.m_step(xi_list, distances_list, B_num, verbose=verbose, verbose_result=verbose_result)

            # Check for convergence
            new_param = self._get_param()
            delta_param = np.abs(old_param - new_param).max()
            if delta_param < self.tolerance:
                # last iteration is not recorded in the loglik_list and param_list
                break
        
        return i, loglik, loglik_list, param_list
    
    def e_step(self, y, alpha, beta):
        """
        Performs the E-step of the EM algorithm with distance-dependent transitions.

        Parameters:
        - y: observation sequence of shape (T,)
        - alpha: forward probabilities of shape (T, n_states)
        - beta: backward probabilities of shape (T, n_states)

        Returns:
        - gamma: state probabilities of shape (T, n_states); gamma(t,i) = P(Q_t=i | Y,params)
        - xi: transition probabilities of shape (T, n_states, n_states); xi(t,i,j) = P(Q_t-1=i, Q_{t}=j | Y,params), first element is 0 matrix
        """

        T, n_states = alpha.shape
        xi = np.zeros((T, n_states, n_states))
        gamma = np.zeros((T, n_states))

        for t in range(1, T):
            A_t = self._transition_tensor[t, :, :]
            numer = (alpha[t-1, :, None] * A_t * self.B[:, y[t]] * beta[t, None, :])
            xi[t] = numer / (numer.sum() + np.finfo(float).eps)
        
        gamma[:-1, :] = np.sum(xi, axis=2)[1:] # use xi to compute the gamma, then compute the last gamma
        gamma[-1, :] = (alpha[-1, :] * beta[-1, :]) / np.dot(alpha[-1, :], beta[-1, :])
        
        return gamma, xi
    
    def m_step(self, xi_list, distances_list, B_num, verbose=True, verbose_result=False):
        """
        Performs the M-step of the EM algorithm.

        Parameters:
        - xi_list: list of transition probabilities of shape (T, n_states, n_states); xi(t,i,j) = P(Q_t-1=i, Q_{t}=j | Y,params), first element is 0 matrix
        - distances_list: list of distance sequences of shape (T,) with first element as 0 (empty)
        - B_num: numerator of emission probabilities, shape (n_states, n_emits), unnormalized probabilities
        - verbose: whether to print optimization progress
        - verbose_result: whether to print optimization result for numeric optimization method
        """

        # Update B with normalization, emission probabilities
        self.B[:]  = B_num / np.sum(B_num, axis=1, keepdims=True)
        
        def objective(trans_param):
            self._update_trans_model(trans_param)
            loss = self._neglog_likelihood(xi_list, distances_list)
            if verbose_result: # not a local variable, access from outer scope
                optim_loss_history.append(loss)
            return loss
        
        optim_loss_history = [] # clean before collect
        init_prob = np.random.random(size=2) 
        init_w    = np.random.normal(size=2) 
        init_w[1] = -np.abs(init_w[1])  # prior knowledge: increase dist -> less dependency, w_1 is negative
        init_guess= np.concatenate([init_prob, init_w])

        result = minimize(objective, init_guess, method='L-BFGS-B', bounds=((0, 1), (0, 1), (None, None), (None, 0)))
        
        if verbose:
            print(f"m-step converge: {result.success}; current loss:{result.fun}; estimates: {np.round(result.x, 4)}")
        if verbose_result:
            print(result)
            print(optim_loss_history)

        # update the model with the new parameters
        self._update_trans_model(result.x)    # might be redundant, but to ensure the model is updated
    
    def _update_model(self, params):
        '''
        Update the model parameters with the given values params = [p1, p2, w0, w1, p3, p4]
        - p1, p2: transition probabilities
        - w0, w1: decay coefficients
        - p3, p4: emission probabilities
        trans_param = params[0:4]
        emits_param = params[-2:]
        '''

        self._update_trans_model(params[0:4])
        self._update_emits_model(params[-2:])
    
    def _update_trans_model(self, trans_param):
        '''
        Update the model parameters with the given values trans_param = [p1, p2, w0, w1]
        the emission probabilities are updated separately in the M-step
        '''

        self.A1[:] = [[1 - trans_param[0], trans_param[0]], [trans_param[1], 1 - trans_param[1]]]
        self.A2[:] = [[trans_param[0], -trans_param[0]], [-trans_param[1], trans_param[1]]]
        #print(f"Before update (id, value): {id(self.w)}, {self.w}")
        #print("Intended update values:", trans_param[2:4])
        self.w[:]  = trans_param[2:4]
        #print(f"After update (id, value): {id(self.w)}, {self.w}")
    
    def _update_emits_model(self, emits_param):
        '''
        Update the emission probabilities with the given values emits_param = [p3, p4]
        '''
        
        self.B[:]  = [[1 - emits_param[0], emits_param[0]], [emits_param[1], 1 - emits_param[1]]]
    
    def _neglog_likelihood(self, xi_list, distances_list):
        '''
        Compute the negative log-likelihood of the model parameters given the observations, which will be the loss to minimize.
        
        Parameters:
        - xi_list: list of transition probabilities of shape (T_r, n_states, n_states); xi(t_r,i,j) = P(Q_t-1=i, Q_{t}=j | Y, params), first element is 0 matrix
        - distances_list: list of distance sequences of shape (T_r,) with first element as 0 (empty)
        T_r is the length of observation sequences in read r

        Returns:
        - total_loglik: negative log-likelihood of the model parameters given the observations
        '''

        total_loglik = 0
        for i in range(len(xi_list)):
            xi = xi_list[i]
            distances = distances_list[i]
            decay_factors, _ = self._get_decay_factors(distances)
            loglik = xi * np.log(self.A1 + self.A2 * decay_factors[:, None, None] + np.finfo(float).eps)
            total_loglik += np.sum(loglik)
        
        return -total_loglik
    
    def predict(self, P_initial, distances, pred_seed=None, verbose=False):
        """
        Generates a sequence of observations for given initial state probability and distances.
        
        Parameters:
        - P_initial: initial state probabilities, shape (n_states,)
        - distances: distance sequence, shape (T,) with first element as 0 (empty)
        - pred_seed: seed for random number generator
        - verbose: whether to print progress

        Returns:
        - y: observed sequence
        - z: state sequence
        """
        
        self.pred_rng = np.random.RandomState(seed=pred_seed)       # self.pred_rng.get_state()[1][0] to get the specified seed

        T = len(distances)
        z = np.zeros(T, dtype=int)
        y = np.zeros(T, dtype=int)
        
        decay_factors, _ = self._get_decay_factors(distances)
        self._transition_tensor = self.A1 + self.A2 * decay_factors[:, None, None]
        # Initial state is chosen based on P_initial, and generate first observation
        z[0] = self.pred_rng.choice(self.n_states, p=P_initial)
        y[0] = self.pred_rng.choice(self.n_emits, p=self.B[z[0], :])
        if verbose:
            print(f"Initial state: {z[0]}; observation: {y[0]}; init_prob: {P_initial}; emit_prob: {self.B};")
        
        # Generate the rest of the sequence
        for t in range(1, T):
            # Transition to next state based on A, generate observation based on B
            A = self._transition_tensor[t, :, :]
            z[t] = self.pred_rng.choice(self.n_states, p=A[z[t-1], :])
            y[t] = self.pred_rng.choice(self.n_emits, p=self.B[z[t], :])
            if verbose:
                print(f"t={t}/{T-1}; state: {z[t]}; observation: {y[t]}; trans_prob: {A}; distance: {distances[t]} ; decay factor: {decay_factors[t]};")
        
        return y, z
    
    def _color(self, text_str, color="blue"):
        """
        Color for state visualization.
        """
        
        color_dict = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", 
                      "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m", "white": "\033[97m"}

        return color_dict[color] + text_str + "\033[0m"
    
    def _get_param(self):
        '''
        Return the model parameters as a 1-D array.(p1, p2, w0, w1, p3, p4)
        '''
        return np.array([self.A1[0,1], self.A1[1,0],
                         self.w[0], self.w[1],
                         self.B[0,1], self.B[1,0]])
