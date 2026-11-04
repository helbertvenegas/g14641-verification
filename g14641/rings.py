"""Coefficient rings.

CycRing(N): Q(zeta_N)[f], with f an indeterminate.  Elements are fmpq_mpoly in
(z, f), kept reduced modulo the cyclotomic polynomial Phi_N(z).  Lex order with
z > f makes the remainder canonical, so "is zero" is an exact test.

GenericRing(): Q[p, h, f], with p, h, f indeterminates.  Used for identities
that hold for all parameters (no root-of-unity assumption).
"""

from fractions import Fraction
from math import gcd

import flint


class CycRing:
    def __init__(self, N):
        if N < 1:
            raise ValueError("N must be positive")
        self.N = N
        self.ctx = flint.fmpq_mpoly_ctx.get(("z", "f"), "lex")
        self.z, self.f = self.ctx.gens()
        cyc = flint.fmpz_poly.cyclotomic(N)
        self.phi_univ = flint.fmpq_poly([int(c) for c in cyc.coeffs()])
        self.phi = self.ctx.from_dict(
            {(i, 0): int(c) for i, c in enumerate(cyc.coeffs()) if c != 0}
        )
        self.zero = self.ctx.from_dict({})
        self.one = self.ctx.from_dict({(0, 0): 1})
        self._zpow = [self.red(self.z ** e) for e in range(N)]

    def red(self, a):
        return a % self.phi

    def zeta(self, e):
        """zeta^e for any integer e."""
        return self._zpow[e % self.N]

    def const(self, c):
        c = Fraction(c)
        return self.ctx.from_dict({(0, 0): flint.fmpq(c.numerator, c.denominator)})

    def is_zero(self, a):
        return a == 0

    # inverse of an element of Q(zeta) (no f)
    def inv(self, a):
        d = a.to_dict()
        if any(k[1] != 0 for k in d):
            raise ValueError("inv only for elements of Q(zeta)")
        coeffs = [0] * (max((k[0] for k in d), default=0) + 1)
        for k, v in d.items():
            coeffs[k[0]] = v
        u = flint.fmpq_poly(coeffs)
        g, s, t = u.xgcd(self.phi_univ)
        if g.degree() != 0:
            raise ZeroDivisionError("not invertible")
        s = s / g[0]
        res = self.ctx.from_dict({(i, 0): s[i] for i in range(s.degree() + 1) if s[i] != 0})
        out = self.red(res)
        assert self.red(out * a) == self.one
        return out

    def __repr__(self):
        return f"Q(zeta_{self.N})[f]"


class GenericRing:
    def __init__(self):
        self.ctx = flint.fmpq_mpoly_ctx.get(("p", "h", "f"), "lex")
        self.p, self.h, self.f = self.ctx.gens()
        self.zero = self.ctx.from_dict({})
        self.one = self.ctx.from_dict({(0, 0, 0): 1})

    def red(self, a):
        return a

    def const(self, c):
        c = Fraction(c)
        return self.ctx.from_dict({(0, 0, 0): flint.fmpq(c.numerator, c.denominator)})

    def is_zero(self, a):
        return a == 0

    def __repr__(self):
        return "Q[p,h,f]"


def order_mod(r, N):
    """Multiplicative order of zeta^r, zeta a primitive N-th root of unity."""
    return N // gcd(r % N, N) if r % N else 1
