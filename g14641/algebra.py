"""Exact arithmetic in G = G(p,f,h) through the PBW normal form.

Elements are dicts {(i, j, k, l): coeff} standing for
    sum coeff * x1^i x2^j y1^k y2^l,
which is a basis of G (Proposition 2.2 of the paper).  Products are computed
from the defining relations only:

    x2 x1 = x1 x2,     y_i x1 = q x1 y_i  (q = hp),     y2 y1 = p y1 y2,
    y1 x2 = h(p^2 x2 y1 + x1 y2),    y2 x2 = h(x2 y2 + f x1 y1).

The last two say (y1, y2)^T x2 = h M (y1, y2)^T with
    M = [[p^2 x2, x1], [f x1, x2]],
so (y1, y2)^T x1^a x2^b = q^a h^b x1^a M^b (y1, y2)^T.  Nothing else from the
paper is used here: every statement checked by the scripts is tested against
this multiplication.
"""

from functools import lru_cache

GENS = ("x1", "x2", "y1", "y2")
_UNIT = {"x1": (1, 0, 0, 0), "x2": (0, 1, 0, 0), "y1": (0, 0, 1, 0), "y2": (0, 0, 0, 1)}


class GAlgebra:
    def __init__(self, ring, p, h, f=None):
        self.R = ring
        red = ring.red
        self.p = red(p)
        self.h = red(h)
        self.f = ring.f if f is None else red(f)
        self.q = red(self.h * self.p)
        self._pow_cache = {}
        self._W = {}
        self._Mpow = {0: [[{(0, 0): ring.one}, {}], [{}, {(0, 0): ring.one}]]}

    # ------------------------------------------------------------------ scalars
    def pw(self, name, e):
        key = (name, e)
        if key not in self._pow_cache:
            base = getattr(self, name)
            if e < 0:
                raise ValueError("negative power")
            v = self.R.one
            for _ in range(e):
                v = self.R.red(v * base)
            self._pow_cache[key] = v
        return self._pow_cache[key]

    # ------------------------------------------------------------- polynomials
    def _polymul(self, A, B):
        out = {}
        red = self.R.red
        for (a1, b1), c1 in A.items():
            for (a2, b2), c2 in B.items():
                k = (a1 + a2, b1 + b2)
                out[k] = out.get(k, self.R.zero) + red(c1 * c2)
        return {k: v for k, v in out.items() if v != 0}

    def _polyadd(self, A, B):
        out = dict(A)
        for k, v in B.items():
            out[k] = out.get(k, self.R.zero) + v
        return {k: v for k, v in out.items() if v != 0}

    def _M(self, j):
        if j not in self._Mpow:
            prev = self._M(j - 1)
            one = self.R.one
            M = [[{(0, 1): self.R.red(self.p * self.p)}, {(1, 0): one}],
                 [{(1, 0): self.f}, {(0, 1): one}]]
            new = [[None, None], [None, None]]
            for r in range(2):
                for s in range(2):
                    acc = {}
                    for t in range(2):
                        acc = self._polyadd(acc, self._polymul(M[r][t], prev[t][s]))
                    new[r][s] = acc
            self._Mpow[j] = new
        return self._Mpow[j]

    # ---------------------------------------------------------------- elements
    def zero(self):
        return {}

    def one(self):
        return {(0, 0, 0, 0): self.R.one}

    def gen(self, name):
        return {_UNIT[name]: self.R.one}

    def mono(self, i, j, k, l, c=None):
        return {(i, j, k, l): self.R.one if c is None else self.R.red(c)}

    def scalar(self, c):
        c = self.R.red(c)
        return {} if c == 0 else {(0, 0, 0, 0): c}

    def add(self, *elts):
        out = {}
        for a in elts:
            for k, v in a.items():
                out[k] = out.get(k, self.R.zero) + v
        return {k: v for k, v in out.items() if v != 0}

    def scale(self, c, a):
        c = self.R.red(c)
        out = {k: self.R.red(c * v) for k, v in a.items()}
        return {k: v for k, v in out.items() if v != 0}

    def sub(self, a, b):
        return self.add(a, self.scale(-self.R.one, b))

    def is_zero(self, a):
        return all(v == 0 for v in a.values())

    # ------------------------------------------------- left mult by a generator
    def _lmul_y(self, r, a):
        """y_r * a, r in {1, 2}."""
        out = {}
        red = self.R.red
        for (i, j, k, l), c in a.items():
            base = red(c * red(self.pw("q", i) * self.pw("h", j)))
            Mj = self._M(j)
            for s in (1, 2):
                poly = Mj[r - 1][s - 1]
                if not poly:
                    continue
                if s == 1:
                    tail, tc = (k + 1, l), self.R.one
                else:
                    tail, tc = (k, l + 1), self.pw("p", k)
                bc = red(base * tc)
                for (a1, b1), pc in poly.items():
                    key = (i + a1, b1, tail[0], tail[1])
                    out[key] = out.get(key, self.R.zero) + red(bc * pc)
        return {k: v for k, v in out.items() if v != 0}

    def lmul_gen(self, g, a):
        if g == "x1":
            return {(i + 1, j, k, l): c for (i, j, k, l), c in a.items()}
        if g == "x2":
            return {(i, j + 1, k, l): c for (i, j, k, l), c in a.items()}
        if g == "y1":
            return self._lmul_y(1, a)
        if g == "y2":
            return self._lmul_y(2, a)
        raise KeyError(g)

    def _Wd(self, k, l, a, b):
        """(y1^k y2^l)(x1^a x2^b) in normal form."""
        key = (k, l, a, b)
        if key not in self._W:
            if k == 0 and l == 0:
                val = {(a, b, 0, 0): self.R.one}
            elif k > 0:
                val = self._lmul_y(1, self._Wd(k - 1, l, a, b))
            else:
                val = self._lmul_y(2, self._Wd(0, l - 1, a, b))
            self._W[key] = val
        return self._W[key]

    # ------------------------------------------------------------------ product
    def mul(self, A, B):
        out = {}
        red = self.R.red
        zero = self.R.zero
        for (i, j, k, l), ca in A.items():
            for (a, b, c, d), cb in B.items():
                cab = red(ca * cb)
                for (i2, j2, k2, l2), cw in self._Wd(k, l, a, b).items():
                    key = (i + i2, j + j2, k2 + c, l2 + d)
                    coef = red(cab * red(cw * self.pw("p", l2 * c)))
                    out[key] = out.get(key, zero) + coef
        return {k: v for k, v in out.items() if v != 0}

    def mul_many(self, *elts):
        out = self.one()
        for a in elts:
            out = self.mul(out, a)
        return out

    def power(self, a, e):
        out = self.one()
        base = a
        while e:
            if e & 1:
                out = self.mul(out, base)
            e >>= 1
            if e:
                base = self.mul(base, base)
        return out

    def comm(self, a, b):
        return self.sub(self.mul(a, b), self.mul(b, a))

    def is_central(self, a):
        return all(self.is_zero(self.comm(a, self.gen(g))) for g in GENS)

    def check_normal(self, a, chi):
        """chi: dict gen -> scalar.  Tests a g = chi(g) g a for the 4 generators."""
        for g in GENS:
            lhs = self.mul(a, self.gen(g))
            rhs = self.scale(chi[g], self.mul(self.gen(g), a))
            if not self.is_zero(self.sub(lhs, rhs)):
                return False
        return True

    # ------------------------------------------------------- weights and forms
    @staticmethod
    def weight(m):
        i, j, k, l = m
        return i + 2 * (j + k + l)

    def leading_form(self, a):
        """Leading form for the filtration deg x1 = 1, deg x2 = deg y_i = 2."""
        if not a:
            return {}
        w = max(self.weight(m) for m in a)
        return {m: c for m, c in a.items() if self.weight(m) == w}

    def total_degrees(self, a):
        return {sum(m) for m in a}


def dickson(alg, lam, a, k):
    """D_k(lam, a) for commuting elements lam, a of the algebra."""
    if k == 0:
        return alg.scalar(alg.R.const(2))
    d0, d1 = alg.scalar(alg.R.const(2)), lam
    for _ in range(k - 1):
        d0, d1 = d1, alg.sub(alg.mul(lam, d1), alg.mul(a, d0))
    return d1
