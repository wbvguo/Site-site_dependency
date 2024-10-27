import argparse
import numpy as np
import pickle

from scipy.optimize import minimize

import sys
sys.path.append('/u/home/w/wbguo/iproject/Site-site_dependency/src/')
from HeterogeneousHiddenMarkovModel import HeterogeneousHiddenMarkovModel


def generate_observations(n, P_initial_list, distances_list, p1, p2, p3, p4, w0, w1):
    '''
    generate n obervations, each observation consists (y, distances, P_initial)
    '''
    estimator = HeterogeneousHiddenMarkovModel()
    estimator.A1  = np.array([[1-p1, p1], [p2, 1-p2]])
    estimator.A2  = np.array([[p1, -p1], [-p2, p2]])
    estimator.B   = np.array([[1-p3, p3], [p4, 1-p4]])
    estimator.w   = [w0, w1]
    
    observations  = []
    for i in range(n):
        P_initial = P_initial_list[i]
        distances = distances_list[i]
        y, z = estimator.predict(P_initial, distances)
        observations.append((y, P_initial, distances))
    return observations


def main(p1, p2, w0, w1, p3, p4, n, outdir, prefix, seed=42):
    random_state = np.random.RandomState(seed=seed)

    # p1 = 0.3
    # p2 = 0.4
    # p3 = 0.05
    # p4 = 0.05
    # w0 = random_state.randn()
    # w1 = -np.abs(random_state.randn())*np.sqrt(1/1000)

    distances_list = [np.hstack((np.array([0]), random_state.randint(1, 1000, size=np.random.randint(5, 10)))) for _ in range(n)]
    P_initial_list = [random_state.dirichlet(alpha = [0.5,0.5]) for _ in range(n)]  # Generate initial probabilities

    observations = generate_observations(n, P_initial_list, distances_list, p1, p2, p3, p4, w0, w1)

    heterhmm = HeterogeneousHiddenMarkovModel(init_seed=42)

    heterhmm.init_param()
    true_params = [p1, p2, w0, w1, p3, p4]
    print(true_params)

    heterhmm.fit(observations,verbose=True,n_starts=20)
    print(heterhmm._get_param())

    with open(f"{outdir}/model_{prefix}.pkl", 'wb') as file:
        pickle.dump([true_params, heterhmm], file)

    with open(f"{outdir}/data_{prefix}.pkl", 'wb') as file:
        pickle.dump(observations, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("-p1", type=float, help="p1", required=True)
    parser.add_argument("-p2", type=float, help="p2", required=True)
    parser.add_argument("-w0", type=float, help="w0", required=True)
    parser.add_argument("-w1", type=float, help="w1", required=True)
    parser.add_argument("-p3", type=float, help="p3", required=True)
    parser.add_argument("-p4", type=float, help="p4", required=True)
    parser.add_argument("-n", type=int, help="number of observations", required=True)
    parser.add_argument("-o", type=str, help="output directory", required=True)
    parser.add_argument("-p", type=str, help="prefix", required=True)
    
    args = parser.parse_args()
    
    main(args.p1, args.p2, args.w0, args.w1, args.p3, args.p4, args.n, args.o, args.p)

