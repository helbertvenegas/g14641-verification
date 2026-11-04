"""Quantum torus on v = (x1, T, y1, y2) at roots of unity.

v_i v_j = zeta^{Omega[i][j]} v_j v_i, so
    v^a v^b = zeta^{sum_{i>j} Omega[i][j] a_i b_j} v^{a+b}.
Elements are dicts {a (tuple of 4 ints): coeff}, coefficients in a CycRing.
"""


class QuantumTorus:
    def __init__(self, ring, Omega):
        self.R = ring
        self.O = Omega

    def scalar(self, c):
        c = self.R.red(c)
        return {} if c == 0 else {(0, 0, 0, 0): c}

    def power(self, x, e):
        out = {(0, 0, 0, 0): self.R.one}
        for _ in range(e):
            out = self.mul(out, x)
        return out

    def mono(self, a, c=None):
        return {tuple(a): self.R.one if c is None else self.R.red(c)}

    def add(self, *elts):
        out = {}
        for x in elts:
            for k, v in x.items():
                out[k] = out.get(k, self.R.zero) + v
        return {k: v for k, v in out.items() if v != 0}

    def scale(self, c, x):
        c = self.R.red(c)
        out = {k: self.R.red(c * v) for k, v in x.items()}
        return {k: v for k, v in out.items() if v != 0}

    def sub(self, x, y):
        return self.add(x, self.scale(-self.R.one, y))

    def _twist(self, a, b):
        e = 0
        for i in range(4):
            for j in range(i):
                e += self.O[i][j] * a[i] * b[j]
        return self.R.zeta(e)

    def mul(self, x, y):
        out = {}
        for a, ca in x.items():
            for b, cb in y.items():
                k = tuple(ai + bi for ai, bi in zip(a, b))
                c = self.R.red(self.R.red(ca * cb) * self._twist(a, b))
                out[k] = out.get(k, self.R.zero) + c
        return {k: v for k, v in out.items() if v != 0}

    def is_zero(self, x):
        return all(v == 0 for v in x.values())

    def commutes_with_gens(self, x):
        for i in range(4):
            e = [0, 0, 0, 0]
            e[i] = 1
            g = self.mono(e)
            if not self.is_zero(self.sub(self.mul(x, g), self.mul(g, x))):
                return False
        return True
