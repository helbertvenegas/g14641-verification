"""Section sec:ozone: ozone groups and homological determinant.

Ozone automorphisms are encoded as (a, t, c) with xi = zeta^a, theta = (-1)^t,
eta = zeta^c.  For the PI case thm:ozall reduces membership in Oz(G) to
eq:ozall on the lattice L; here we compute that group from a Z-basis of L
(Smith normal form) and compare it, element by element, with the explicit
lists of cor:ozcases and Theorem B.

We also test the lattice description against the algebra itself: an element
of the group must fix the central lifts Lambda_m (m in a Hilbert basis of S),
and the conjugations by the normal elements of cor:ozcases (i) must be the
automorphisms stated there.
"""

import os
import sys
from math import gcd

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.lattice import (abelian_order_statistics, admissible, hilbert_basis, lattice_L,
                            order_statistics, ozone_group, pi_degree_formula)
from g14641.report import Report, finish
from g14641.setup import Setup


def as_set(G):
    return set(G)


def expected_ozcases(n, case):
    """Explicit lists of cor:ozcases, encoded over zeta with N as returned."""
    out = set()
    if case == "h=p^-1":
        N = n
        if n % 2:
            out = {(0, 0, 0)}
        else:
            m = n // 2
            # sigma = phi_{1,1,-1}, tau = phi_{1,-1,1}; -1 = zeta^m
            out = {(0, 0, 0), (0, 0, m), (0, 1, 0), (0, 1, m)}
    elif case == "h=1":
        N = n
        for a in range(n):
            for c in range(n):
                for t in (0, 1):
                    # xi^n = eta^n = 1 automatically; (theta xi)^n or ^m = 1
                    k = n if n % 2 else n // 2
                    # (theta xi)^k = (-1)^{tk} zeta^{ak}
                    if (2 * a * k + t * k * n) % (2 * n) == 0:
                        out.add((a, t, c))
    elif case == "h=p":
        N = n
        d = n // gcd(n, 4)
        g = gcd(n, 2 * d)
        for a in range(n):
            for c in range(n):
                for t in (0, 1):
                    xi_ok = (a * d) % n == 0
                    eta_ok = (c * g) % n == 0
                    th_ok = (t * n) % 2 == 0
                    if xi_ok and eta_ok and th_ok:
                        out.add((a, t, c))
    elif case == "h=-p^-1":
        if n % 2 == 0:
            N, m = n, n // 2
            for t in (0, 1):
                for ce in (0, m):
                    # xi = theta^m
                    a = (t * m * (n // 2)) % n  # theta^m = (-1)^{tm} = zeta^{(n/2) tm}
                    out.add((a, t, ce))
        else:
            N = 2 * n
            for a in (0, n):
                for c in (0, n):
                    out.add((a, 0, c))
    return N, out


def params(n, case):
    if case == "h=p^-1":
        return n, 1, n - 1
    if case == "h=1":
        return n, 1, 0
    if case == "h=p":
        return n, 1, 1
    if case == "h=-p^-1":
        if n % 2 == 0:
            return n, 1, n // 2 - 1
        return 2 * n, 2, n - 2
    raise KeyError(case)


def expected_iso(n, case):
    if case == "h=p^-1":
        return [1] if n % 2 else [2, 2]
    if case == "h=1":
        return [n, n] if n % 4 else [n // 2, 2, n]
    if case == "h=p":
        return [n, n] if n % 4 else [n // 4, 2, n // 2]
    if case == "h=-p^-1":
        return [2, 2]


def check_cases(rep, nmax=40):
    for n in range(3, nmax + 1):
        for case in ("h=p^-1", "h=1", "h=p", "h=-p^-1"):
            N, r, s = params(n, case)
            assert admissible(N, r, s)
            G = ozone_group(N, r, s)
            Nexp, expect = expected_ozcases(n, case)
            tag = f"n={n},{case}"
            rep.check("cor:ozcases explicit list", tag, Nexp == N and as_set(G) == expect,
                      f"|computed|={len(G)}, |expected|={len(expect)}")
            rep.check("cor:ozcases isomorphism type", tag,
                      order_statistics(G, N) == abelian_order_statistics(expected_iso(n, case)))
            pd = pi_degree_formula(N, r, s)
            rep.check("rem:ozCY |Oz| divides rank = PIdeg^2", tag, (pd * pd) % len(G) == 0)
            rep.check("rem:ozCY |Oz| < rank (so G is not a skew polynomial ring)", tag, len(G) < pd * pd)
            if case in ("h=p^-1", "h=1") and n % 2:
                want = 1 if case == "h=p^-1" else n * n
                rep.check("thm:B special cases (n odd)", tag, len(G) == want and
                          (case == "h=p^-1" or order_statistics(G, N) == abelian_order_statistics([n, n])))


def check_general(rep, Nmax=30):
    for N in range(3, Nmax + 1):
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                tag = f"N={N},r={r},s={s}"
                G = ozone_group(N, r, s)
                # Nakayama automorphism phi_{q^-2, 1, q^2}
                qe = r + s
                nak = ((-2 * qe) % N, 0, (2 * qe) % N)
                rep.check("rem:ozCY nu_G in Oz(G)", tag, nak in G)
                # thm:hdetCY (i) <=> (iii): hdet = xi^2 eta^2
                hdet_trivial = all((2 * a + 2 * c) % N == 0 for (a, t, c) in G)
                cy = (2 * qe) % N == 0
                rep.check("thm:hdetCY (i) <=> (iii)", tag, hdet_trivial == cy)
                # abelian: automatic; finite: |Oz| divides PIdeg^2
                pd = pi_degree_formula(N, r, s)
                rep.check("rem:ozCY |Oz| divides PIdeg^2 (general)", tag, (pd * pd) % len(G) == 0)


def check_against_algebra(rep, Nmax=6):
    """For small N: phi in Oz (lattice) <=> phi fixes every Lambda_m, m in the
    Hilbert basis of S.  phi multiplies a PBW monomial x^m by
    xi^{m1+m2} theta^{m2+m4} eta^{m3+m4}; we apply it to the actual elements."""
    for N in range(3, Nmax + 1):
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                tag = f"N={N},r={r},s={s}"
                st = Setup(N, r, s)
                A, R = st.alg, st.R
                lifts = [st.Lambda(m) for m in hilbert_basis(N, r, s)]
                G = set(ozone_group(N, r, s))
                ok = True
                for a in range(N):
                    for c in range(N):
                        for t in (0, 1):
                            fixes = True
                            for L in lifts:
                                img = {}
                                for (i, j, k, l), coef in L.items():
                                    sc = R.red(R.zeta(a * (i + j) + c * (k + l)) * R.const((-1) ** (t * (j + l))))
                                    img[(i, j, k, l)] = R.red(sc * coef)
                                if not A.is_zero(A.sub(img, L)):
                                    fixes = False
                                    break
                            if fixes != ((a, t, c) in G):
                                ok = False
                rep.check("thm:ozall graded part: fixes Z(G) <=> eq:ozall", tag, ok)


def check_conjugations(rep):
    """cor:ozcases (i), n = 2m: sigma, tau, sigma tau are conjugation by Q_m, y1^m, y2^m."""
    for n in (4, 6, 8, 10):
        m = n // 2
        st = Setup(n, 1, n - 1)
        A = st.alg
        tag = f"n={n},h=p^-1"
        # a g = chi(g) g a; conjugation g -> a^-1 g a = chi(g)^{-1}... chi = +-1 here
        cases = {"Q_m -> sigma=phi_{1,1,-1}": (st.Q(m), (0, 0, m, m)),
                 "y1^m -> tau=phi_{1,-1,1}": (A.power(st.y1, m), (0, m, 0, m)),
                 "y2^m -> sigma tau=phi_{1,-1,-1}": (A.power(st.y2, m), (0, m, m, 0))}
        for name, (a, chi) in cases.items():
            rep.check("cor:ozcases (i) conjugation by normal elements", tag + ": " + name,
                      A.check_normal(a, st.chi(chi)) and not A.is_central(a))


def check_hdet(rep):
    t, xi, eta, th = sp.symbols("t xi eta theta")
    for thv in (1, -1):
        Tr = 1 / ((1 - xi * t) * (1 - thv * xi * t) * (1 - eta * t) * (1 - thv * eta * t))
        u = sp.symbols("u")  # u = 1/t
        ser = sp.series(Tr.subs(t, 1 / u), u, 0, 6).removeO()
        lead = sp.simplify(ser.coeff(u, 4))
        low = all(sp.simplify(ser.coeff(u, k)) == 0 for k in range(0, 4))
        rep.check("lem:hdet trace expansion: leading term xi^-2 eta^-2 t^-4", f"theta={thv}",
                  low and sp.simplify(lead - 1 / (xi ** 2 * eta ** 2)) == 0)
    # phi(x1 C) = xi^2 eta^2 x1 C: every monomial of x1 C has i+j = 2, k+l = 2, j+l even
    for (N, r, s) in [(5, 1, 3), (6, 1, 0), (8, 2, 1), (7, 3, 2)]:
        st = Setup(N, r, s)
        x1C = st.alg.mul(st.x1, st.C)
        ok = all(i + j == 2 and k + l == 2 and (j + l) % 2 == 0 for (i, j, k, l) in x1C)
        rep.check("lem:hdet phi(x1 C) = xi^2 eta^2 x1 C", f"N={N},r={r},s={s}", ok)
    # thm:hdetCY (i) <=> (ii): x1 C central iff q^2 = 1
    for N in range(3, 9):
        for r in range(1, N):
            for s in range(N):
                if admissible(N, r, s):
                    st = Setup(N, r, s)
                    rep.check("thm:hdetCY (i) <=> (ii): x1 C central iff q^2 = 1", f"N={N},r={r},s={s}",
                              st.alg.is_central(st.alg.mul(st.x1, st.C)) == ((2 * (r + s)) % N == 0))


def main():
    rep = Report("05 ozone groups (sec:ozone)")
    check_cases(rep)
    check_general(rep)
    check_conjugations(rep)
    check_hdet(rep)
    check_against_algebra(rep)
    finish(rep)


if __name__ == "__main__":
    main()
