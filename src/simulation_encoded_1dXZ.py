from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz
MAX_ITS = 100000

def array2state(L, arr):
    """
    Convert a numpy array to an L-qubit state.
    """
    assert 2**L == len(arr)
    s = State(L=L)
    s.set_all_by_function(lambda i: arr[i])
    return s

def get_encoder_isometry(nb):
    # logical states per block
    v00 = np.sqrt(0.5) * (State(state='0001') - State(state='1110'))
    v10 = np.sqrt(0.5) * (State(state='0100') - State(state='1011'))
    v01 = np.sqrt(0.5) * (State(state='1101') - State(state='0010'))
    v11 = np.sqrt(0.5) * (State(state='1000') - State(state='0111'))
    v00_np = v00.to_numpy()
    v10_np = v10.to_numpy()
    v01_np = v01.to_numpy()
    v11_np = v11.to_numpy()
    # encoder isometry per block
    U0 = sp.sparse.csc_matrix(np.column_stack((v00_np, v10_np, v01_np, v11_np)))
    # overall encoder isometry
    Uenc = functools.reduce(sp.sparse.kron, [U0] * nb)
    return Uenc

if __name__ == '__main__':
    # ----- parse command line input -----
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--nb", type=int, required=True,
                           help="number of blocks")
    argparser.add_argument("--noise", type=float, required=True,
                           help="noise strength")
    argparser.add_argument("--lamb", type=int, required=True,
                           help="penalty coefficient")
    argparser.add_argument("--time", type=float, required=True,
                           help="total evolving time")
    argparser.add_argument("--nt", type=int, required=True,
                           help="number of time steps")
    argparser.add_argument("--seed", type=int, required=True,
                           help="random seed (used to sample the noise coefficients and the initial state)")
    args = argparser.parse_args()

    nb = args.nb
    noise = args.noise

    if args.lamb >= 0:
        lamb_list = [args.lamb]
    elif args.lamb == -1:
        lamb_list = 2 ** np.arange(5,15)
    else:
        print("invalid lamb")
        exit(1)

    t_list = (args.time / args.nt) * np.arange(1, args.nt + 1)

    if args.seed >= 0:
        seed_list = [args.seed]
    elif args.seed == -1:
        seed_list = list(range(20))
    else:
        print("invalid seed")
        exit(1)

    # ----- build target Hamiltonian `Htar`, penalty Hamiltonian `Hpen`, encoding Hamiltonian `Henc1` & `Henc2`, encoder isometry `Uenc` -----
    nl = 2 * nb # number of logical qubits
    n = 4 * nb # number of physical qubits
    
    Htar = index_sum(X(0) * X(1), size=nl) + index_sum(Z(0) * Z(1), size=nl)
    Htar.L = nl
    
    Hpen = sum([X(2*i) * X(2*i+1) + 3 * Z(2*i) * Z(2*i+1) for i in range(2 * nb)])
    Hpen.L = n

    Henc1 = sum([Z(4*i+1)*Z(4*i+2) - X(4*i+1)*X(4*i+2) for i in range(nb)]) # inner-block logical ZZ & XX
    Henc1 += sum([-3.5*Z(4*i+4)*Z(4*i+5) + 1.5*X(4*i)*X(4*i+1) for i in range(nb-1)]) # compensate for residual terms from cross-block gadget
    Henc1.L = n
    Henc2 = sum([np.sqrt(8/3) * (Z(4*i+1)*X(4*i+6) + Z(4*i+3)*X(4*i+6) + 3*Z(4*i+1)*X(4*i+4)) for i in range(nb-1)]) # cross-block logical ZZ & XX
    Henc2.L = n

    Uenc = get_encoder_isometry(nb)
    Penc = Uenc @ Uenc.conj().T

    for lamb in lamb_list:
        for seed in seed_list:
            for t in t_list:
                # build simulator Hamiltonian
                Hsim = lamb * Hpen + Henc1 + np.sqrt(lamb) * Henc2
            
                # add coherent noise
                np.random.seed(seed)
                epsx = noise * np.random.uniform(-1,1,n)
                epsy = noise * np.random.uniform(-1,1,n)
                epsz = noise * np.random.uniform(-1,1,n)
                V = sum([epsx[i] * X(i) + epsy[i] * Y(i) + epsz[i] * Z(i) for i in range(n)])
                V.L = n
            
                # build noisy total Hamiltonian
                Htot = Hsim + V
            
                # randomly select an initial state
                psi0 = State(L=nl, state='random', seed=seed) # lower case: unencoded states
                psi0_np = psi0.to_numpy()
                PSI0_np = Uenc @ psi0_np
                PSI0 = array2state(L=n, arr=PSI0_np) # UPPER case: encoded states
                
                # evolve
                psi = Htar.evolve(psi0, t=t)
                PSI = Htot.evolve(PSI0, t=t, max_its=MAX_ITS)
                psi_np = psi.to_numpy()
                PSI_np = PSI.to_numpy()
                Uenc_psi_np = Uenc @ psi_np

                # analyze
                innerprod = np.dot(PSI_np.conj(), Uenc_psi_np)
                Penc_PSI_np = Penc @ PSI_np
                Penc_PSI_norm = np.linalg.norm(Penc_PSI_np)
                leakage = 1 - Penc_PSI_norm ** 2
                postsel_PSI_np = Penc_PSI_np / Penc_PSI_norm
                postsel_innerprod = np.dot(postsel_PSI_np.conj(), Uenc_psi_np)

                # output
                f = open(f"output_1dXZ_{nb}blocks_noise={noise}_seed={seed}.txt", "a")
                f.write(f"#blocks = {nb}, noise = {noise}, lamb = {lamb}, t = {t}, seed = {seed}, innerprod = {innerprod}, leakage = {leakage}, postsel_innerprod = {postsel_innerprod}\n")
                f.close()