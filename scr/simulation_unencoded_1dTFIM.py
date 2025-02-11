from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", type=int, required=True,
                           help="number of qubits")
    argparser.add_argument("--noise", type=float, required=True,
                           help="noise strength")
    argparser.add_argument("-t", type=float, required=True,
                           help="evolving time")
    argparser.add_argument("--seed", type=int, required=True,
                           help="random seed (used to sample the noise coefficients and the initial state)")
    args = argparser.parse_args()

    if args.seed >= 0:
        seed_list = [args.seed]
    elif args.seed == -50:
        seed_list = list(range(50))
    elif args.seed == -20:
        seed_list = list(range(20))
    else:
        print("invalid seed")
        exit(1)

    if args.t > 0:
        t_list = [args.t]
    elif args.t == -10:
        t_list = 0.1 * np.arange(1,101)
    elif args.t == -1:
        t_list = 0.02 * np.arange(1,51)
    else:
        print("invalid t")
        exit(1)
    
    n = args.n
    noise = args.noise

    for t in t_list:
        for seed in seed_list:

            # build Hamiltonian
            H = index_sum(X(0), size=n) + index_sum(Z(0), size=n) + index_sum(Z(0) * Z(1), size=n)
            H.L = n

            # add coherent error
            np.random.seed(seed)
            epsx = noise * np.random.uniform(-1,1,n)
            epsy = noise * np.random.uniform(-1,1,n)
            epsz = noise * np.random.uniform(-1,1,n)
            V = sum([epsx[i] * X(i) + epsy[i] * Y(i) + epsz[i] * Z(i) for i in range(n)])
            V.L = n

            # build noisy Hamiltonian
            H_noisy = H + V

            # randomly select an initial state
            psi0 = State(L=n, state='random', seed=seed)

            # evolve
            psi_true = H.evolve(psi0, t=t)
            psi_noisy = H_noisy.evolve(psi0, t=t)
            psi_true_np = psi_true.to_numpy()
            psi_noisy_np = psi_noisy.to_numpy()
            innerprod = np.dot(psi_noisy_np.conj(), psi_true_np)

            # output
            f = open(f"sweep_time_{n}qubits_noise={noise}.txt", "a")
            f.write(f"#qubits = {n}, noise = {noise}, t = {t}, seed = {seed}, innerprod = {innerprod}\n")
            f.close()