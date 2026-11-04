"""Section sec:birational: the birational model.

Inside the quantum torus on (x1, T, y1, y2) with the relations eq:qas we put
    rho = y1 y2^-1,  beta = y1 y2,  u = b rho + c rho^-1,  x2 := x1 (T + u)
and check that x1, x2, y1, y2 satisfy the defining relations eq:G.  This is
the map phi : G -> Q(P) in the proof of thm:birational; Lemma lem:inner is the
statement that T = x1^-1 x2 - u has the relations eq:T, which hold in the
torus by construction.

Two versions: generic p, h, f (sympy, Laurent coefficients) and all
admissible roots of unity with N <= 12 (exact, f generic).
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.algebra import dickson
from g14641.lattice import (admissible, in_L, in_L_paper_congruences, omega_from_relations,
                            omega_paper, order)
from g14641.report import Report, finish
from g14641.rings import CycRing
from g14641.torus import QuantumTorus

X1, T, Y1, Y2 = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)


def vadd(*vs):
    return tuple(sum(c) for c in zip(*vs))


def vneg(v):
    return tuple(-c for c in v)


# ---------------------------------------------------------------- generic case
class SymTorus:
    """Quantum torus with v_i v_j = p^{Op_ij} q^{Oq_ij} v_j v_i, sympy coefficients."""

    def __init__(self, p, q):
        self.p, self.q = p, q
        Op = [[0] * 4 for _ in range(4)]
        Oq = [[0] * 4 for _ in range(4)]

        def put(M, i, j, e):
            M[i][j], M[j][i] = e, -e

        put(Oq, 2, 0, 1); put(Oq, 3, 0, 1)          # y_i x1 = q x1 y_i
        put(Op, 1, 2, -1); put(Op, 1, 3, 1)         # T y1 = p^-1 y1 T, T y2 = p y2 T
        put(Op, 3, 2, 1)                            # y2 y1 = p y1 y2
        self.Op, self.Oq = Op, Oq

    def mul(self, x, y):
        out = {}
        for a, ca in x.items():
            for b, cb in y.items():
                ep = sum(self.Op[i][j] * a[i] * b[j] for i in range(4) for j in range(i))
                eq = sum(self.Oq[i][j] * a[i] * b[j] for i in range(4) for j in range(i))
                k = vadd(a, b)
                out[k] = out.get(k, 0) + ca * cb * self.p ** ep * self.q ** eq
        return out

    def add(self, *xs):
        out = {}
        for x in xs:
            for k, v in x.items():
                out[k] = out.get(k, 0) + v
        return out

    def scale(self, c, x):
        return {k: c * v for k, v in x.items()}

    def is_zero(self, x):
        return all(sp.simplify(sp.together(v)) == 0 for v in x.values())


def generic_checks(rep):
    p, h, f = sp.symbols("p h f", nonzero=True)
    q = h * p
    S = SymTorus(p, q)
    b, c = f / (p ** 2 - 1), -1 / (p ** 2 - 1)
    m = lambda v: {v: sp.Integer(1)}
    x1, Tm, y1, y2 = m(X1), m(T), m(Y1), m(Y2)
    rho = S.mul(y1, m(vneg(Y2)))
    rhoi = S.mul(y2, m(vneg(Y1)))
    u = S.add(S.scale(b, rho), S.scale(c, rhoi))
    t = S.add(Tm, u)
    x2 = S.mul(x1, t)
    rels = {
        "x2 x1 = x1 x2": S.add(S.mul(x2, x1), S.scale(-1, S.mul(x1, x2))),
        "y1 x1 = hp x1 y1": S.add(S.mul(y1, x1), S.scale(-q, S.mul(x1, y1))),
        "y2 x1 = hp x1 y2": S.add(S.mul(y2, x1), S.scale(-q, S.mul(x1, y2))),
        "y2 y1 = p y1 y2": S.add(S.mul(y2, y1), S.scale(-p, S.mul(y1, y2))),
        "y1 x2 = h(p^2 x2 y1 + x1 y2)": S.add(S.mul(y1, x2), S.scale(-h * p ** 2, S.mul(x2, y1)),
                                             S.scale(-h, S.mul(x1, y2))),
        "y2 x2 = h(x2 y2 + f x1 y1)": S.add(S.mul(y2, x2), S.scale(-h, S.mul(x2, y2)),
                                           S.scale(-h * f, S.mul(x1, y1))),
    }
    for name, val in rels.items():
        rep.check("thm:birational: x2 := x1(T+u) satisfies eq:G", f"generic p,h,f: {name}", S.is_zero(val))
    # eq:t: y1 t = p t y1 + p^-1 y2, y2 t = p^-1 t y2 + f p^-1 y1 (h has disappeared)
    e1 = S.add(S.mul(y1, t), S.scale(-p, S.mul(t, y1)), S.scale(-1 / p, y2))
    e2 = S.add(S.mul(y2, t), S.scale(-1 / p, S.mul(t, y2)), S.scale(-f / p, y1))
    rep.check("eq:t", "generic p,h,f", S.is_zero(e1) and S.is_zero(e2))
    # lem:inner: delta_0(r) = u r - sigma_0(r) u
    d1 = S.add(S.mul(u, y1), S.scale(-1 / p, S.mul(y1, u)), S.scale(p ** -2, y2))
    d2 = S.add(S.mul(u, y2), S.scale(-p, S.mul(y2, u)), S.scale(f, y1))
    rep.check("lem:inner: delta_0 is inner, implemented by u", "generic p,h,f", S.is_zero(d1) and S.is_zero(d2))
    # eq:C: C = x1 T beta, with C = x2 y1 y2 - x1(c y2^2 + p^-1 b y1^2)
    beta = S.mul(y1, y2)
    C = S.add(S.mul(S.mul(x2, y1), y2),
              S.scale(-c, S.mul(x1, S.mul(y2, y2))), S.scale(-b / p, S.mul(x1, S.mul(y1, y1))))
    rep.check("eq:C: C = x1 T beta in Q(G)", "generic p,h,f",
              S.is_zero(S.add(C, S.scale(-1, S.mul(S.mul(x1, Tm), beta)))))
    # rem:CGWZ17: x2' = x1 T normalizes with sigma
    x2p = S.mul(x1, Tm)
    ok = (S.is_zero(S.add(S.mul(y1, x2p), S.scale(-h * p ** 2, S.mul(x2p, y1))))
          and S.is_zero(S.add(S.mul(y2, x2p), S.scale(-h, S.mul(x2p, y2)))))
    rep.check("rem:CGWZ17: x2' = x2 - x1 u is sigma-normal", "generic p,h,f", ok)


# ----------------------------------------------------------- roots of unity
def root_checks(rep, Nmax=12):
    for N in range(3, Nmax + 1):
        R = CycRing(N)
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                case = f"N={N},r={r},s={s}"
                O = omega_from_relations(r, s)
                rep.check("eq:Omega matrix as printed = relations eq:qas", case, O == omega_paper(r, s))
                Tq = QuantumTorus(R, O)
                p, h = R.zeta(r), R.zeta(s)
                q = R.red(p * h)
                den = R.inv(R.red(p * p - R.one))
                b, c = R.red(R.f * den), R.red(-den)
                m = Tq.mono
                x1, Tm, y1, y2 = m(X1), m(T), m(Y1), m(Y2)
                rho = Tq.mul(y1, m(vneg(Y2)))
                rhoi = Tq.mul(y2, m(vneg(Y1)))
                u = Tq.add(Tq.scale(b, rho), Tq.scale(c, rhoi))
                t = Tq.add(Tm, u)
                x2 = Tq.mul(x1, t)
                sc = Tq.scale
                rels = [
                    Tq.sub(Tq.mul(x2, x1), Tq.mul(x1, x2)),
                    Tq.sub(Tq.mul(y1, x1), sc(q, Tq.mul(x1, y1))),
                    Tq.sub(Tq.mul(y2, x1), sc(q, Tq.mul(x1, y2))),
                    Tq.sub(Tq.mul(y2, y1), sc(p, Tq.mul(y1, y2))),
                    Tq.sub(Tq.mul(y1, x2), sc(h, Tq.add(sc(R.red(p * p), Tq.mul(x2, y1)), Tq.mul(x1, y2)))),
                    Tq.sub(Tq.mul(y2, x2), sc(h, Tq.add(Tq.mul(x2, y2), sc(R.f, Tq.mul(x1, y1))))),
                ]
                rep.check("thm:birational: x2 := x1(T+u) satisfies eq:G", case, all(Tq.is_zero(x) for x in rels))

                # eq:rhobeta for k <= 2N
                beta = Tq.mul(y1, y2)
                ok = True
                for k in range(1, 2 * N + 1):
                    rk, bk = Tq.power(rho, k), Tq.power(beta, k)
                    ok &= Tq.is_zero(Tq.sub(rk, m((0, 0, k, -k), R.zeta(-r * k * (k - 1) // 2))))
                    ok &= Tq.is_zero(Tq.sub(bk, m((0, 0, k, k), R.zeta(r * k * (k - 1) // 2))))
                    ok &= Tq.is_zero(Tq.sub(Tq.mul(rk, bk), m((0, 0, 2 * k, 0), R.zeta(-r * k * k))))
                    rki = Tq.power(rhoi, k)
                    ok &= Tq.is_zero(Tq.sub(Tq.mul(rki, bk), m((0, 0, 0, 2 * k), R.zeta(r * k * (k - 1)))))
                rep.check("eq:rhobeta", case, ok)

                # lem:normal (iv): Q_e = x1^e (T^e + b^e rho^e + c^e rho^-e), pairwise commuting
                e = order(2 * r, N)
                alpha = R.red(b * c)
                Qe = dickson(Tq, x2, sc(alpha, Tq.mul(x1, x1)), e)
                be, ce = R.one, R.one
                for _ in range(e):
                    be, ce = R.red(be * b), R.red(ce * c)
                Te, re_, rie = Tq.power(Tm, e), Tq.power(rho, e), Tq.power(rhoi, e)
                rhs = Tq.mul(Tq.power(x1, e), Tq.add(Te, sc(be, re_), sc(ce, rie)))
                comm = (Tq.is_zero(Tq.sub(Tq.mul(Te, re_), Tq.mul(re_, Te)))
                        and Tq.is_zero(Tq.sub(Tq.mul(Te, rie), Tq.mul(rie, Te))))
                rep.check("lem:normal (iv): Q_e = x1^e(T^e + b^e rho^e + c^e rho^-e)", case,
                          Tq.is_zero(Tq.sub(Qe, rhs)) and comm)

                # prop:rationalcenter: congruences eq:congruences <=> Omega a = 0 mod N
                ok = True
                for i in range(N):
                    for j in range(N):
                        for k in range(N):
                            for l in range(N):
                                a = (i, j, k, l)
                                if in_L(N, r, s, a) != in_L_paper_congruences(N, r, s, a):
                                    ok = False
                rep.check("prop:rationalcenter eq:congruences = rows of Omega a = 0", case, ok)

                # eq:conjtorus / eq:Lmult: M_a central in the torus <=> a in L (sample)
                if N <= 6:
                    ok = True
                    for i in range(N):
                        for j in range(N):
                            for k in range(N):
                                for l in range(N):
                                    a = (i, j, k, l)
                                    if Tq.commutes_with_gens(m(a)) != in_L(N, r, s, a):
                                        ok = False
                    rep.check("prop:torus / eq:Lmult: M_a central <=> a in L", case, ok)


def main():
    rep = Report("02 birational model (sec:birational)")
    generic_checks(rep)
    root_checks(rep)
    finish(rep)


if __name__ == "__main__":
    main()
