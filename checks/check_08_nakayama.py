"""prop:CY: the Nakayama automorphism, through the Koszul dual.

G is Koszul AS-regular of dimension 4, so E = G^! = T(V*)/(R^perp) is
Frobenius with Hilbert series (1+t)^4, and for a Koszul AS-regular algebra
of dimension d,
    nu_G |_{G_1} = (-1)^{d+1} (nu_E |_{E_1})^*        (Van den Bergh, Smith).
For k[x1..x4] this gives E = exterior algebra and nu_E = -id on E_1.

We compute E_1, E_3, E_4 and nu_E on E_1 by linear algebra over F_P
(zeta -> primitive N-th root of unity mod P, f -> f0), and check that nu_E is
diagonal on x1*, x2*, y1*, y2* with eigenvalues
    -q^{-2}, -q^{-2}, -q^{2}, -q^{2}     (or the inverses, depending on
                                          the order of the pairing),
which is what eq:nakayama predicts.  In particular nu_G = id iff q^2 = 1.
This is a check over F_P, not over Q(zeta): it confirms eq:nakayama for one
reduction of the parameters.
"""

import os
import sys
from itertools import product

import flint
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.lattice import admissible
from g14641.report import Report, finish


def prime_1_mod(N, start=10 ** 9):
    P = start - (start % N) + 1
    while not sp.isprime(P):
        P += N
    return P


def prim_root(P, N):
    from g14641.setup import _primes
    for cand in range(2, P):
        w = pow(cand, (P - 1) // N, P)
        if all(pow(w, N // pr, P) != 1 for pr in _primes(N)):
            return w


def nakayama_dual(N, r, s, f0=3):
    P = prime_1_mod(N)
    w = prim_root(P, N)
    z = lambda e: pow(w, e % N, P)
    p, h = z(r), z(s)
    q = p * h % P
    f = f0 % P
    X1, X2, Y1, Y2 = range(4)
    # relations in V (x) V, as dicts {(a, b): coeff}
    rels = [
        {(X2, X1): 1, (X1, X2): -1},
        {(Y1, X1): 1, (X1, Y1): -q},
        {(Y2, X1): 1, (X1, Y2): -q},
        {(Y2, Y1): 1, (Y1, Y2): -p},
        {(Y1, X2): 1, (X2, Y1): -h * p * p, (X1, Y2): -h},
        {(Y2, X2): 1, (X2, Y2): -h, (X1, Y1): -h * f},
    ]
    Rm = flint.nmod_mat(6, 16, P)
    for i, rel in enumerate(rels):
        for (a, b), c in rel.items():
            Rm[i, 4 * a + b] = c % P
    Rperp_mat, dim = Rm.nullspace()
    assert dim == 10
    Rperp = [[int(Rperp_mat[k, j]) for k in range(16)] for j in range(dim)]

    def ideal_part(k):
        gens = []
        for i in range(k - 1):
            for left in product(range(4), repeat=i):
                for right in product(range(4), repeat=k - 2 - i):
                    for v in Rperp:
                        vec = [0] * (4 ** k)
                        for ab in range(16):
                            if v[ab]:
                                a, b = divmod(ab, 4)
                                word = left + (a, b) + right
                                idx = 0
                                for x in word:
                                    idx = 4 * idx + x
                                vec[idx] = (vec[idx] + v[ab]) % P
                        gens.append(vec)
        return gens

    I3 = ideal_part(3)
    I4 = ideal_part(4)
    M4 = flint.nmod_mat(len(I4), 256, P)
    for i, v in enumerate(I4):
        for j, c in enumerate(v):
            if c:
                M4[i, j] = c
    eps_mat, d4 = M4.nullspace()
    rank4 = M4.rank()
    M3 = flint.nmod_mat(len(I3), 64, P)
    for i, v in enumerate(I3):
        for j, c in enumerate(v):
            if c:
                M3[i, j] = c
    rref3, rank3 = M3.rref()
    dims_ok = (rank3 == 60 and rank4 == 255 and d4 == 1)
    eps = [int(eps_mat[j, 0]) for j in range(256)]
    # basis of E_3: words that are not pivots of rref3
    pivots = set()
    for i in range(rank3):
        for j in range(64):
            if int(rref3[i, j]):
                pivots.add(j)
                break
    free = [j for j in range(64) if j not in pivots]
    if len(free) != 4:
        return None, dims_ok, (P, w)

    def word3(j):
        return (j // 16, (j // 4) % 4, j % 4)

    def epsw(word):
        idx = 0
        for x in word:
            idx = 4 * idx + x
        return eps[idx]

    B = flint.nmod_mat(4, 4, P)
    B2 = flint.nmod_mat(4, 4, P)
    for i in range(4):
        for jj, j in enumerate(free):
            B[i, jj] = epsw((i,) + word3(j))
            B2[jj, i] = epsw(word3(j) + (i,))
    Nmat = B2.inv() * B.transpose()
    return Nmat, dims_ok, (P, w)


def main(Nmax=12):
    rep = Report("08 Nakayama automorphism through the Koszul dual (prop:CY)")
    # control: q = 1 is the CY case; we also treat every admissible triple
    for N in range(3, Nmax + 1):
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                tag = f"N={N},r={r},s={s}"
                Nmat, dims_ok, (P, w) = nakayama_dual(N, r, s)
                rep.check("G^! has dims 4, 4, 1 in degrees 1, 3, 4 (Koszul dual of AS-regular dim 4)", tag, dims_ok)
                if Nmat is None:
                    rep.check("prop:CY eq:nakayama via G^!", tag, False, "E_3 basis")
                    continue
                q = pow(w, (r + s) % N, P)
                qi = pow(q, P - 2, P)
                off = all(int(Nmat[i, j]) == 0 for i in range(4) for j in range(4) if i != j)
                d = [int(Nmat[i, i]) for i in range(4)]
                opt1 = [(-qi * qi) % P] * 2 + [(-q * q) % P] * 2
                opt2 = [(-q * q) % P] * 2 + [(-qi * qi) % P] * 2
                rep.check("prop:CY eq:nakayama via G^!: nu diagonal, eigenvalues -q^{-+2} on x*, -q^{+-2} on y*",
                          tag, off and (d == opt1 or d == opt2), f"diag={d}")
                cy = all(x == (P - 1) for x in d)
                rep.check("prop:CY Calabi-Yau iff h = +-p^-1 (q^2 = 1)", tag, cy == ((2 * (r + s)) % N == 0))
    finish(rep)


if __name__ == "__main__":
    main()
