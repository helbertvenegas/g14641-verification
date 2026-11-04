"""Integer and lattice computations for G(p,f,h) at roots of unity.

Parameters are encoded as in Section 3 of the paper: zeta is a primitive N-th
root of unity, N = |<p, h>|, p = zeta^r, h = zeta^s, gcd(r, s, N) = 1.
"""

from functools import lru_cache
from itertools import product
from math import gcd

import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_decomp


def admissible(N, r, s):
    """N = |<zeta^r, zeta^s>| and p^2 != 1."""
    return gcd(gcd(r % N, s % N), N) == 1 and (2 * r) % N != 0


def order(e, N):
    e %= N
    return N // gcd(e, N) if e else 1


def omega_from_relations(r, s):
    """Commutation exponents of the quantum torus on v = (x1, T, y1, y2),
    read off from the relations (3.6) of the paper:
        T x1 = x1 T,  y_i x1 = hp x1 y_i,  T y1 = p^-1 y1 T,  T y2 = p y2 T,
        y2 y1 = p y1 y2.
    v_i v_j = zeta^{Omega[i][j]} v_j v_i."""
    O = [[0] * 4 for _ in range(4)]

    def put(i, j, e):  # v_i v_j = zeta^e v_j v_i
        O[i][j] = e
        O[j][i] = -e

    put(1, 0, 0)          # T x1 = x1 T
    put(2, 0, r + s)      # y1 x1 = hp x1 y1
    put(3, 0, r + s)      # y2 x1 = hp x1 y2
    put(1, 2, -r)         # T y1 = p^-1 y1 T
    put(1, 3, r)          # T y2 = p y2 T
    put(3, 2, r)          # y2 y1 = p y1 y2
    return O


def omega_paper(r, s):
    """Matrix (3.7) as printed in the paper."""
    return [[0, 0, -(r + s), -(r + s)],
            [0, 0, -r, r],
            [r + s, r, 0, -r],
            [r + s, -r, r, 0]]


@lru_cache(maxsize=None)
def lattice_L(N, r, s):
    """Returns (basis, index) with basis a list of 4 vectors generating
    L = {a in Z^4 : Omega a = 0 mod N} and index = [Z^4 : L]."""
    M = Matrix(omega_from_relations(r, s))
    D, U, V = smith_normal_decomp(M, domain=ZZ)
    assert U * M * V == D and abs(U.det()) == 1 and abs(V.det()) == 1
    d = [int(D[i, i]) for i in range(4)]
    scale = [N // gcd(N, di) for di in d]
    basis = []
    for i in range(4):
        col = [int(V[k, i]) * scale[i] for k in range(4)]
        basis.append(tuple(col))
    index = 1
    for x in scale:
        index *= x
    for b in basis:  # sanity: basis vectors lie in L
        assert in_L(N, r, s, b)
    return tuple(basis), index


def in_L(N, r, s, a):
    O = omega_from_relations(r, s)
    return all(sum(O[i][j] * a[j] for j in range(4)) % N == 0 for i in range(4))


def in_L_paper_congruences(N, r, s, a):
    """The congruences (3.10) of Proposition 3.6, as printed."""
    i, j, k, l = a
    return ((r + s) * (k + l) % N == 0 and r * (l - k) % N == 0
            and ((r + s) * i + r * (j - l)) % N == 0
            and ((r + s) * i + r * (k - j)) % N == 0)


def image_size_bruteforce(N, r, s):
    O = np.array(omega_from_relations(r, s), dtype=np.int64)
    pts = np.array(list(product(range(N), repeat=4)), dtype=np.int64)
    img = (pts @ O.T) % N
    return len({tuple(x) for x in img.tolist()})


def pi_degree_formula(N, r, s):
    return N * N // gcd(N, 2 * r * (r + s))


# --------------------------------------------------------------------------
# The graded lattice and the semigroup S = L_gr cap N^4
# --------------------------------------------------------------------------

def L_gr_basis(N, r, s):
    """m in L_gr iff (m1+m2, m2, m3, m4) in L."""
    basis, _ = lattice_L(N, r, s)
    return tuple((i - j, j, k, l) for (i, j, k, l) in basis)


def box_points_in_Lgr(N, r, s, lo=0, hi=None):
    """All m in [lo, hi)^4 with m in L_gr (numpy array)."""
    hi = N if hi is None else hi
    O = np.array(omega_from_relations(r, s), dtype=np.int64)
    rng = np.arange(lo, hi, dtype=np.int64)
    m1, m2, m3, m4 = np.meshgrid(rng, rng, rng, rng, indexing="ij")
    m = np.stack([m1.ravel(), m2.ravel(), m3.ravel(), m4.ravel()], axis=1)
    a = m.copy()
    a[:, 0] = m[:, 0] + m[:, 1]
    ok = np.all((a @ O.T) % N == 0, axis=1)
    return m[ok]


def hilbert_numerator(N, r, s):
    """k[S] has Hilbert series P(t)/(1-t^N)^4 for the total degree, with
    P(t) = sum of t^{|m|} over m in S with all m_c < N."""
    pts = box_points_in_Lgr(N, r, s)
    deg = pts.sum(axis=1)
    P = np.bincount(deg, minlength=4 * N)
    return [int(x) for x in P]


def series_from_numerator(P, N, upto):
    """Coefficients of P(t)/(1-t^N)^4 up to degree upto."""
    out = [0] * (upto + 1)
    # 1/(1-t^N)^4 = sum C(a+3,3) t^{Na}
    for i, c in enumerate(P):
        if c == 0:
            continue
        a = 0
        while i + N * a <= upto:
            out[i + N * a] += c * ((a + 1) * (a + 2) * (a + 3) // 6)
            a += 1
    return out


def is_palindromic(P):
    nz = [i for i, c in enumerate(P) if c]
    lo, hi = nz[0], nz[-1]
    core = P[lo:hi + 1]
    return core == core[::-1]


def hilbert_basis(N, r, s):
    """Hilbert basis of S.  It lies in [0, N)^4 together with the N e_c:
    any m in S with m_c >= N is N e_c + (m - N e_c), both in S."""
    pts = [tuple(x) for x in box_points_in_Lgr(N, r, s).tolist()]
    pts = [x for x in pts if any(x)]
    pts.sort(key=sum)
    S_small = set(pts)
    hb = []
    for x in pts:
        reducible = False
        for y in hb:
            d = tuple(a - b for a, b in zip(x, y))
            if min(d) >= 0 and any(d) and (d in S_small):
                reducible = True
                break
        if not reducible:
            hb.append(x)
    for c in range(4):
        e = [0, 0, 0, 0]
        e[c] = N
        e = tuple(e)
        # N e_c is irreducible unless it is a sum inside the box
        red = False
        for y in hb:
            d = tuple(a - b for a, b in zip(e, y))
            if min(d) >= 0 and any(d) and (d in S_small or d == e):
                red = True
                break
        if not red:
            hb.append(e)
    return hb


# --------------------------------------------------------------------------
# Ozone groups (Theorem 4.4): phi_{xi,theta,eta} with xi = zeta^a,
# eta = zeta^c, theta = (-1)^t.  Encoded as triples (a, t, c).
# --------------------------------------------------------------------------

def ozone_group(N, r, s):
    basis, _ = lattice_L(N, r, s)
    gens = list(basis)
    out = []
    for a in range(N):
        for c in range(N):
            for t in (0, 1):
                ok = True
                for (i, j, k, l) in gens:
                    if (2 * (a * i + c * (k + l)) + t * (j + l) * N) % (2 * N):
                        ok = False
                        break
                if ok:
                    out.append((a, t, c))
    return out


def element_order(x, N):
    a, t, c = x
    o = 1
    while True:
        if (o * a) % N == 0 and (o * c) % N == 0 and (o * t) % 2 == 0:
            return o
        o += 1


def order_statistics(group, N):
    stats = {}
    for x in group:
        o = element_order(x, N)
        stats[o] = stats.get(o, 0) + 1
    return dict(sorted(stats.items()))


def abelian_order_statistics(cyclic_orders):
    """Order statistics of Z/m1 x ... x Z/mk."""
    stats = {}
    for x in product(*[range(m) for m in cyclic_orders]):
        o = 1
        for xi, mi in zip(x, cyclic_orders):
            oi = mi // gcd(xi, mi) if xi else 1
            o = o * oi // gcd(o, oi)
        stats[o] = stats.get(o, 0) + 1
    return dict(sorted(stats.items()))
