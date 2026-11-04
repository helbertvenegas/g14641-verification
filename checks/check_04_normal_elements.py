"""Section sec:normal and the Dickson lemmas of sec:prelim.

  prop:dickson, lem:qdickson     Dickson identities
  lem:grmon                      gr G and its central monomials
  lem:normal (ii), (iii)         normal elements and their characters
  paragraph after lem:normal     which normal elements are central
  lem:finite                     the central subalgebra R
"""

import os
import random
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.algebra import GENS, dickson
from g14641.lattice import admissible, in_L, order
from g14641.report import Report, finish
from g14641.rings import CycRing
from g14641.setup import Setup
from g14641.torus import QuantumTorus


def dickson_sym(k, lam, a):
    d0, d1 = sp.Integer(2), lam
    if k == 0:
        return d0
    for _ in range(k - 1):
        d0, d1 = d1, sp.expand(lam * d1 - a * d0)
    return d1


def check_dickson(rep):
    lam, a, w, mu = sp.symbols("lambda a w mu")
    for k in range(0, 16):
        Dk = dickson_sym(k, lam, a)
        ok1 = sp.simplify(sp.expand(Dk.subs(lam, w + a / w)) - (w ** k + a ** k / w ** k)) == 0
        ok2 = sp.expand(dickson_sym(2 * k, lam, a) - (Dk ** 2 - 2 * a ** k)) == 0
        ok3 = sp.expand(dickson_sym(k, mu * lam, mu ** 2 * a) - mu ** k * Dk) == 0
        ok4 = k == 0 or sp.expand(Dk.subs(a, 0) - lam ** k) == 0
        rep.check("prop:dickson eq:dicksonw", f"k={k}", ok1)
        rep.check("prop:dickson eq:dickson2k", f"k={k}", ok2)
        rep.check("prop:dickson eq:dicksonhom", f"k={k}", ok3 and ok4)
    # eq:prodroots for q a primitive k-th root of unity, A', B' symbolic
    import flint
    for k in range(2, 13):
        ctx = flint.fmpq_mpoly_ctx.get(("z", "l", "A", "B"), "lex")
        z, L, Ap, Bp = ctx.gens()
        cyc = flint.fmpz_poly.cyclotomic(k)
        phi = ctx.from_dict({(i, 0, 0, 0): int(c) for i, c in enumerate(cyc.coeffs()) if c != 0})
        prod = ctx.from_dict({(0, 0, 0, 0): 1})
        for j in range(k):
            zj = (z ** j) % phi
            zmj = (z ** ((-j) % k)) % phi
            prod = (prod * (L - Ap * zj - Bp * zmj)) % phi
        d0, d1 = 2 * ctx.from_dict({(0, 0, 0, 0): 1}), L
        for _ in range(k - 1):
            d0, d1 = d1, L * d1 - Ap * Bp * d0
        rhs = d1 - Ap ** k - Bp ** k
        rep.check("lem:qdickson eq:prodroots", f"k={k}", (prod - rhs) % phi == 0)


def check_qdickson(rep):
    """D_k(T + a1 r + a2 r^-1, a1 a2) = T^k + a1^k r^k + a2^k r^-k when T r = q r T,
    q a primitive k-th root of unity.  a1 = f (indeterminate) and a2 runs over
    k+1 rational values; both sides have degree <= k in a2, so the identity
    holds for a2 indeterminate as well."""
    for k in range(2, 11):
        R = CycRing(k)
        O = [[0] * 4 for _ in range(4)]
        O[0][1], O[1][0] = 1, -1          # v0 = T, v1 = r:  T r = zeta r T
        Tq = QuantumTorus(R, O)
        T = Tq.mono((1, 0, 0, 0))
        r = Tq.mono((0, 1, 0, 0))
        ri = Tq.mono((0, -1, 0, 0))
        ok = True
        for a2v in range(-1, k + 1):
            a1, a2 = R.f, R.const(a2v if a2v else 7)
            lam = Tq.add(T, Tq.scale(a1, r), Tq.scale(a2, ri))
            lhs = dickson(Tq, lam, Tq.scalar(R.red(a1 * a2)), k)
            a1k, a2k = R.one, R.one
            for _ in range(k):
                a1k, a2k = R.red(a1k * a1), R.red(a2k * a2)
            rhs = Tq.add(Tq.power(T, k), Tq.scale(a1k, Tq.power(r, k)), Tq.scale(a2k, Tq.power(ri, k)))
            ok &= Tq.is_zero(Tq.sub(lhs, rhs))
        rep.check("lem:qdickson eq:qdickson", f"k={k}", ok)


def gr_mul_scalar(st, a, b):
    """Twist scalar for x^a x^b in gr G (quantum affine space of lem:grmon(i)),
    as a power of zeta."""
    r, s = st.r, st.s
    qe = r + s
    # v = (x1, x2, y1, y2); v_i v_j = zeta^{O[i][j]} v_j v_i for i > j
    O = {(1, 0): 0, (2, 0): qe, (3, 0): qe, (2, 1): s + 2 * r, (3, 1): s, (3, 2): r}
    e = 0
    for i in range(4):
        for j in range(i):
            e += O[(i, j)] * a[i] * b[j]
    return e


def check_grmon(rep, st, case):
    A = st.alg
    R = st.R
    random.seed(3)
    ok = True
    for _ in range(25):
        el = []
        for d in (random.randint(1, 4), random.randint(1, 4)):
            x = {}
            for _ in range(3):
                m = [0, 0, 0, 0]
                for _ in range(d):
                    m[random.randrange(4)] += 1
                x[tuple(m)] = R.const(random.randint(1, 4))
            el.append(x)
        a, b = el
        lab = A.leading_form(A.mul(a, b))
        la, lb = A.leading_form(a), A.leading_form(b)
        prod = {}
        for ma, ca in la.items():
            for mb, cb in lb.items():
                k = tuple(x + y for x, y in zip(ma, mb))
                prod[k] = prod.get(k, R.zero) + R.red(R.red(ca * cb) * R.zeta(gr_mul_scalar(st, ma, mb)))
        prod = {k: v for k, v in prod.items() if v != 0}
        if not A.is_zero(A.sub(lab, prod)):
            ok = False
    rep.check("lem:grmon (i): lf(ab) = lf(a) lf(b) in the quantum affine space", case, ok)

    N, r, s = st.N, st.r, st.s
    ok = True
    for m1 in range(N):
        for m2 in range(N):
            for m3 in range(N):
                for m4 in range(N):
                    m = (m1, m2, m3, m4)
                    central = all(gr_mul_scalar(st, m, g) % N == gr_mul_scalar(st, g, m) % N
                                  for g in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])
                    if central != in_L(N, r, s, (m1 + m2, m2, m3, m4)):
                        ok = False
    rep.check("lem:grmon (ii): central monomials of gr G <-> L", case, ok)

    e, ell = st.e, st.ell
    ok = True
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if in_L(N, r, s, (i, j, k, l)):
                        if not ((j - k) % e == 0 and (k - l) % e == 0 and (2 * i * (r + s)) % N == 0):
                            ok = False
    rep.check("lem:grmon (iii): (i,j,k,l) in L => j=k=l mod e, q^{2i}=1", case, ok)


def check_normal(rep, st, case):
    A, R = st.alg, st.R
    N, r, s, n, e = st.N, st.r, st.s, st.n, st.e
    x1, x2, y1, y2 = st.x1, st.x2, st.y1, st.y2

    # lem:normal (ii): D_k(M, alpha x1^2) = p^k Q_k I for p^{2k} = 1, M as in sec:normal
    pinv = R.zeta(-r)
    M = [[A.scale(st.p, x2), A.scale(pinv, x1)], [A.scale(R.red(R.f * pinv), x1), A.scale(pinv, x2)]]
    a = A.scale(st.alpha, A.mul(x1, x1))

    def mm(P, Q):
        return [[A.add(A.mul(P[i][0], Q[0][j]), A.mul(P[i][1], Q[1][j])) for j in range(2)] for i in range(2)]

    def ms(c, P):
        return [[A.mul(c, P[i][j]) for j in range(2)] for i in range(2)]

    def madd(P, Q, sgn=1):
        return [[A.add(P[i][j], A.scale(R.const(sgn), Q[i][j])) for j in range(2)] for i in range(2)]

    two = A.scalar(R.const(2))
    I2 = [[two, {}], [{}, two]]
    D0, D1 = I2, M
    Ds = {0: D0, 1: D1}
    for k in range(2, 2 * e + 1):
        D0, D1 = D1, madd(mm(M, D1), ms(a, D0), -1)
        Ds[k] = D1
    for k in (e, 2 * e):
        pk = R.zeta(r * k)
        Qk = st.Q(k)
        target = A.scale(pk, Qk)
        ok = (A.is_zero(A.sub(Ds[k][0][0], target)) and A.is_zero(A.sub(Ds[k][1][1], target))
              and A.is_zero(Ds[k][0][1]) and A.is_zero(Ds[k][1][0]))
        rep.check("lem:normal (ii): D_k(M, alpha x1^2) = p^k Q_k I", case + f",k={k}", ok)

    # lem:normal (iii): the table of characters
    rep.check("lem:normal (iii) table: x1", case, A.check_normal(x1, st.chi(st.chi_exp("x1"))))
    for k in (e, 2 * e):
        rep.check("lem:normal (iii) table: Q_k", case + f",k={k}", A.check_normal(st.Q(k), st.chi(st.chi_exp("Q", k))))
    rep.check("lem:normal (iii) table: C", case, A.check_normal(st.C, st.chi(st.chi_exp("C"))))
    for k in (e, 2 * e):
        rep.check("lem:normal (iii) table: y1^r", case + f",r={k}", A.check_normal(A.power(y1, k), st.chi(st.chi_exp("y1", k))))
        rep.check("lem:normal (iii) table: y2^r", case + f",r={k}", A.check_normal(A.power(y2, k), st.chi(st.chi_exp("y2", k))))

    # paragraph after lem:normal
    ell = st.ell
    rep.check("after lem:normal: x1^{ord q} central", case, A.is_central(A.power(x1, ell)))
    hn1 = (s * n) % N == 0
    rep.check("after lem:normal: Y1, Y2, U central iff h^n = 1", case,
              all(A.is_central(z) == hn1 for z in (st.Y(1), st.Y(2), st.U())))
    if n % 2 == 0:
        m = n // 2
        rep.check("after lem:normal: x1^m central iff q^m = 1", case,
                  A.is_central(A.power(x1, m)) == ((r + s) * m % N == 0))
        rep.check("after lem:normal: Q_m central iff h^m = 1", case,
                  A.is_central(st.Q(m)) == ((s * m) % N == 0))
        ym = A.mul(A.power(y1, m), A.power(y2, m))
        anti = all(A.is_zero(A.add(A.mul(ym, g), A.mul(g, ym))) for g in (y1, y2))
        rep.check("after lem:normal: y1^m y2^m anticommutes with y1, y2", case, anti)


def check_finite(rep, st, case):
    A = st.alg
    N, e = st.N, st.e
    zs = [A.power(st.x1, N), A.power(st.Q(e), N), A.power(st.y1, N), A.power(st.y2, N)]
    rep.check("lem:finite: R = k[x1^N, Q_e^N, y1^N, y2^N] is central", case, all(A.is_central(z) for z in zs))
    lfs = [A.leading_form(z) for z in zs]
    expect = [(N, 0, 0, 0), (0, e * N, 0, 0), (0, 0, N, 0), (0, 0, 0, N)]
    rep.check("lem:finite: leading forms x1^N, x2^{eN}, y1^N, y2^N", case,
              all(list(l.keys()) == [m] for l, m in zip(lfs, expect)))


def main(Nmax=10):
    rep = Report("04 normal elements and Dickson polynomials (sec:normal)")
    check_dickson(rep)
    check_qdickson(rep)
    for N in range(3, Nmax + 1):
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                case = f"N={N},r={r},s={s}"
                st = Setup(N, r, s)
                if N <= 6:
                    check_grmon(rep, st, case)
                check_normal(rep, st, case)
                if N <= 5:
                    check_finite(rep, st, case)
    finish(rep)


if __name__ == "__main__":
    main()
