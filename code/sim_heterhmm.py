import argparse
import numpy as np
import pickle

import sys
from pathlib import Path
sys.path.append(f'{Path.home()}/iproject/Site-site_dependency/src/')
from HeterogeneousHMM import HeterogeneousHMM
from SyntheticDataGenerator import generate_sequences_heterhmm


def main(p1, p2, w0, w1, p3, p4, n, outdir, prefix, seed=42):
    random_state = np.random.RandomState(seed=seed)

    distances_list = [np.hstack((np.array([0]), random_state.randint(1, 200, size=np.random.randint(5, 10)))) for _ in range(n)]
    P_initial_list = [random_state.dirichlet(alpha = [0.5,0.5]) for _ in range(n)]  # Generate initial probabilities
    observations   = generate_sequences_heterhmm(n, P_initial_list, distances_list, p1, p2, w0, w1, p3, p4)
    
    true_params  = [p1, p2, w0, w1, p3, p4]
    print(true_params)

    heterhmm = HeterogeneousHMM(init_seed=42)
    heterhmm.fit(observations,verbose=True,n_starts=20)
    print(heterhmm._get_params())

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
