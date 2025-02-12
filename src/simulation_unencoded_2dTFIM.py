from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--nr", type=int, required=True,
                           help="number of rows of qubits")
    argparser.add_argument("--nc", type=int, required=True,
                           help="number of columns of qubits")
    argparser.add_argument("--noise", type=float, required=True,
                           help="noise strength")
    argparser.add_argument("--time", type=float, required=True,
                           help="total evolving time")
    argparser.add_argument("--nt", type=int, required=True,
                           help="number of time steps")
    argparser.add_argument("--seed", type=int, required=True,
                           help="random seed (used to sample the noise coefficients and the initial state)")
    args = argparser.parse_args()

    n_row = args.nr
    n_col = args.nc
    n = n_row * n_col # number of qubits
    noise = args.noise
    t_list = (args.time / args.nt) * np.arange(1, args.nt + 1)

    if args.seed >= 0:
        seed_list = [args.seed]
    elif args.seed == -1:
        seed_list = list(range(20))
    else:
        print("invalid seed")
        exit(1)

    def idx(c, r):
        """
        Return the index for the qubit located at column `c` and row `r`.
        `c`, `r`, and the returned index all start from 0.
        """
        assert 0 <= c and c < n_col
        assert 0 <= r and r < n_row
        return c * n_row + r
    
    # ----- build Hamiltonian -----
    H = 0
    for c in range(n_col):
        for r in range(n_row):
            H += Z(idx(c, r)) + X(idx(c, r))
            if r < n_row - 1:
                H += Z(idx(c, r)) * Z(idx(c, r+1))
            if c < n_col - 1:
                H += Z(idx(c, r)) * Z(idx(c+1, r))
    H.L = n
    
    for t in t_list:
        for seed in seed_list:
            # add coherent noise
            np.random.seed(seed)
            epsx = noise * np.random.uniform(-1,1,n)
            epsy = noise * np.random.uniform(-1,1,n)
            epsz = noise * np.random.uniform(-1,1,n)
            V = sum([epsx[i] * X(i) + epsy[i] * Y(i) + epsz[i] * Z(i) for i in range(n)])
            V.L = n

            # build noisy total Hamiltonian
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
            f = open(f"output_unencoded_sweep_time_2dTFIM_{n_row}x{n_col}_noise={noise}_seed={seed}.txt", "a")
            f.write(f"#rows = {n_row}, #columns = {n_col}, noise = {noise}, t = {t}, seed = {seed}, innerprod = {innerprod}\n")
            f.close()