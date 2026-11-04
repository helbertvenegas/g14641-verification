"""The data of Sections 3-6 of the paper for one choice of (N, r, s):
p = zeta^r, h = zeta^s, zeta a primitive N-th root of unity, f generic.

Everything named here is defined exactly as in the paper:
    b = f/(p^2-1),  c = -1/(p^2-1),  alpha = bc,
    n = ord(p),  e = ord(p^2),  epsilon = p^e,  q = hp,
    Q_k = D_k(x2, alpha x1^2),  U = Q_n,  Y_i = y_i^n,
    C = x2 y1 y2 - x1 (c y2^2 + p^-1 b y1^2).
"""

from functools import lru_cache

import flint

from .algebra import GAlgebra, dickson
from .lattice import order
from .rings import CycRing


class FpRing:
    """F_P with a fixed primitive N-th root of unity omega and f = f0.
    Reduction Z[zeta, f] -> F_P along zeta -> omega, f -> f0."""

    def __init__(self, P, N, f0):
        if (P - 1) % N:
            raise ValueError("need N | P-1")
        self.P, self.N = P, N
        g = None
        for cand in range(2, P):
            w = pow(cand, (P - 1) // N, P)
            if all(pow(w, N // pr, P) != 1 for pr in _primes(N)):
                g = w
                break
        self.omega = g
        self.one = flint.nmod(1, P)
        self.zero = flint.nmod(0, P)
        self.f = flint.nmod(f0, P)

    def red(self, a):
        return a

    def zeta(self, e):
        return flint.nmod(pow(self.omega, e % self.N, self.P), self.P)

    def const(self, c):
        from fractions import Fraction
        c = Fraction(c)
        return flint.nmod(c.numerator, self.P) / flint.nmod(c.denominator, self.P)

    def inv(self, a):
        return self.one / a


def _primes(n):
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


class Setup:
    def __init__(self, N, r, s, ring=None):
        self.N, self.r, self.s = N, r, s
        self.R = CycRing(N) if ring is None else ring
        R = self.R
        self.p = R.zeta(r)
        self.h = R.zeta(s)
        self.alg = GAlgebra(R, self.p, self.h)
        A = self.alg
        self.f = A.f
        self.q = A.q
        self.n = order(r, N)
        self.e = order(2 * r, N)
        self.ell = order(r + s, N)          # ord(q)
        self.eps = R.zeta(r * self.e)       # p^e = +-1
        den = R.inv(R.red(self.p * self.p - R.one))
        self.b = R.red(A.f * den)
        self.c = R.red(-den)
        self.alpha = R.red(self.b * self.c)
        self.x1, self.x2 = A.gen("x1"), A.gen("x2")
        self.y1, self.y2 = A.gen("y1"), A.gen("y2")
        pinv = R.zeta(-r)
        self.C = A.sub(
            A.mul_many(self.x2, self.y1, self.y2),
            A.mul(self.x1, A.add(A.scale(self.c, A.power(self.y2, 2)),
                                 A.scale(R.red(pinv * self.b), A.power(self.y1, 2)))))
        self._Q = {}

    def z(self, e):
        return self.R.zeta(e)

    def Q(self, k):
        if k not in self._Q:
            A = self.alg
            a = A.scale(self.alpha, A.power(self.x1, 2))
            self._Q[k] = dickson(A, self.x2, a, k)
        return self._Q[k]

    def U(self):
        return self.Q(self.n)

    def Y(self, i):
        return self.alg.power(self.y1 if i == 1 else self.y2, self.n)

    # characters of Lemma 4.5(iii), as exponents of zeta on (x1, x2, y1, y2)
    def chi_exp(self, name, k=None):
        r, s = self.r, self.s
        qe = r + s
        if name == "x1":
            return (0, 0, -qe, -qe)
        if name == "Q":
            return (0, 0, -s * k, -s * k)
        if name == "C":
            return (2 * qe, 2 * qe, -qe, -qe)
        if name == "y1":
            return (qe * k, qe * k + r * k, 0, -r * k)
        if name == "y2":
            return (qe * k, qe * k - r * k, r * k, 0)
        raise KeyError(name)

    def chi(self, exps):
        return dict(zip(("x1", "x2", "y1", "y2"), (self.z(e) for e in exps)))

    # central lift Lambda_m of (5.2)
    def Lambda(self, m):
        m1, m2, m3, m4 = m
        e = self.e
        j0 = m2 % e
        sig = (m2 - j0) // e
        assert (m3 - j0) % e == 0 and (m4 - j0) % e == 0 and m3 >= j0 and m4 >= j0
        A = self.alg
        return A.mul_many(A.power(self.Q(e), sig), A.power(self.x1, m1),
                          A.power(self.C, j0), A.power(self.y1, m3 - j0),
                          A.power(self.y2, m4 - j0))
