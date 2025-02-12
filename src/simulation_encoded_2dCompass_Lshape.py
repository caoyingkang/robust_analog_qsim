from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz
MAX_ITS = 100000

flavor_gx = [1, 2, 3]
flavor_gz = [4, 5, 6]

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

    def lidx(c, r, i):
        """
        Return the logical qubit index for the `i`-th logical qubit of the block located at column `c` and row `r`.
        `c`, `r`, and the returned index all start from 0.
        `i` takes value from {1,2}.
        """
        assert (c, r) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        assert i == 1 or i == 2
        return c * 6 + r * 2 + i - 1
    
    def pidx(c, r, i):
        """
        Return the physical qubit index for the `i`-th physical qubit of the block located at column `c` and row `r`.
        `c`, `r`, and the returned index all start from 0.
        `i` takes value from {1,2,3,4}.
        """
        assert (c, r) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        assert i == 1 or i == 2 or i == 3 or i == 4
        return c * 12 + r * 4 + i - 1
    
    # ----- build target Hamiltonian `Htar`, penalty Hamiltonian `Hpen`, encoding Hamiltonian `Henc1` & `Henc2`, encoder isometry `Uenc` -----
    nb = 5 # number of blocks
    nl = 2 * nb # number of logical qubits
    n = 4 * nb # number of physical qubits

    Htar = 0
    for c, r in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        i1, i2 = lidx(c, r, 1), lidx(c, r, 2)
        Htar += X(i1) * X(i2) # horizontal inner-block XX at block (c,r)
        
        if (c, r) in [(0, 0), (0, 1), (1, 0)]: # vertical cross-block ZZ between block (c,r) and block (c,r+1)
            u1, u2, d1, d2 = lidx(c, r, 1), lidx(c, r, 2), lidx(c, r + 1, 1), lidx(c, r + 1, 2)
            Htar += Z(u1) * Z(d2) + Z(u2) * Z(d1)

        if (c, r) in [(0, 0), (0, 1)]: # horizontal cross-block XX between block (c,r) and block (c+1,r)
            l1, l2, r1, r2 = lidx(c, r, 1), lidx(c, r, 2), lidx(c + 1, r, 1), lidx(c + 1, r, 2)
            if r % 2 == 0:
                Htar += X(l2) * X(r1)
            else:
                Htar += X(l1) * X(r2)
    Htar.L = nl

    Hpen = 0
    for c, r in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        gx, gz = flavor_gx[r], flavor_gz[r]
        i1, i2, i3, i4 = pidx(c, r, 1), pidx(c, r, 2), pidx(c, r, 3), pidx(c, r, 4)
        Hpen += gx * X(i1) * X(i2) + gz * Z(i1) * Z(i2) + gx * X(i3) * X(i4) + gz * Z(i3) * Z(i4)
    Hpen.L = n

    Henc1 = 0
    Henc2 = 0
    for c, r in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]:
        i1, i2, i3, i4 = pidx(c, r, 1), pidx(c, r, 2), pidx(c, r, 3), pidx(c, r, 4)
        Henc1 += - X(i2) * X(i3) # horizontal inner-block XX at block (c,r)

        if (c, r) in [(0, 0), (0, 1), (1, 0)]: # vertical cross-block ZZ between block (c,r) and block (c,r+1)
            u1, u2, u3, u4 = pidx(c, r, 1), pidx(c, r, 2), pidx(c, r, 3), pidx(c, r, 4)
            d1, d2, d3, d4 = pidx(c, r + 1, 1), pidx(c, r + 1, 2), pidx(c, r + 1, 3), pidx(c, r + 1, 4)
            
            gx, gz = flavor_gx[(r+1) % 3], flavor_gz[r % 3]
            alpha = np.sqrt((gz ** 2 - gx ** 2) / gz)
            Henc2 += alpha * (X(u3) * Z(d2) + X(u3) * Z(d4))
            Henc1 += Z(u1) * Z(u2) # compensate for residual terms from cross-block gadget

            gx, gz = flavor_gx[r % 3], flavor_gz[(r+1) % 3]
            alpha = np.sqrt((gz ** 2 - gx ** 2) / gz)
            Henc2 += alpha * (X(d3) * Z(u2) + X(d3) * Z(u4))
            Henc1 += Z(d1) * Z(d2) # compensate for residual terms from cross-block gadget

        if (c, r) in [(0, 0), (0, 1)]: # horizontal cross-block XX between block (c,r) and block (c+1,r)
            l1, l2, l3, l4 = pidx(c, r, 1), pidx(c, r, 2), pidx(c, r, 3), pidx(c, r, 4)
            r1, r2, r3, r4 = pidx(c + 1, r, 1), pidx(c + 1, r, 2), pidx(c + 1, r, 3), pidx(c + 1, r, 4)
            
            gx, gz = flavor_gx[r % 3], flavor_gz[r % 3]
            alpha = np.sqrt((gz ** 2 - gx ** 2) / gx)
            if r % 2 == 0:
                Henc2 += alpha * (Z(l2) * X(r1) + Z(l2) * X(r3))
                Henc1 += X(l1) * X(l2) # compensate for residual terms from cross-block gadget
            else:
                Henc2 += alpha * (Z(r2) * X(l1) + Z(r2) * X(l3))
                Henc1 += X(r1) * X(r2) # compensate for residual terms from cross-block gadget
    Henc1.L = n
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
                f = open(f"output_2dCompass_Lshape_noise={noise}_seed={seed}.txt", "a")
                f.write(f"#blocks = {nb}, noise = {noise}, lamb = {lamb}, t = {t}, seed = {seed}, innerprod = {innerprod}, leakage = {leakage}, postsel_innerprod = {postsel_innerprod}\n")
                f.close()