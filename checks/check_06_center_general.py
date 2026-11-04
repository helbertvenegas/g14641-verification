"""Section sec:general: the center in the PI case.

  lem:Lambda      the lifts Lambda_m are central with leading form x^m
  thm:gencenter   (i)  gr Z(G) = Z(gr G) = k[S]: dim Z(G)_d = #S_d for small d
                  (iii) Gorenstein <=> h in <-1,p>
  lem:proj, lem:arith
  lem:Ai, thm:qhyp (i)  the quantum hypersurface relation eq:Frel

For (i) the lower bound dim Z(G)_d >= #S_d comes from the lifts Lambda_m,
checked exactly with f an indeterminate.  The upper bound is the kernel of
a -> ([a,x1],[a,x2],[a,y1],[a,y2]) on G_d, computed after reducing
Z[zeta, f] -> F_P (zeta -> a primitive N-th root of unity mod P, f -> f0).
Reduction can only lower the rank, so this kernel bounds dim Z(G)_d from
above for f an indeterminate and for f = f0.
"""

import os
import sys
from itertools import product
from math import comb, gcd

import flint
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.algebra import GENS, GAlgebra
from g14641.lattice import (L_gr_basis, admissible, box_points_in_Lgr, hilbert_basis,
                            hilbert_numerator, in_L, is_palindromic, order, series_from_numerator)
from g14641.report import Report, finish
from g14641.setup import FpRing, Setup


def prime_1_mod(N, start=10 ** 9):
    P = start - (start % N) + 1
    while not sp.isprime(P):
        P += N
    return P


def monomials(d):
    out = []
    for i in range(d + 1):
        for j in range(d + 1 - i):
            for k in range(d + 1 - i - j):
                out.append((i, j, k, d - i - j - k))
    return out


def center_dims_modP(N, r, s, D, f0=3):
    P = prime_1_mod(N)
    R = FpRing(P, N, f0)
    A = GAlgebra(R, R.zeta(r), R.zeta(s))
    gens = [A.gen(g) for g in GENS]
    dims = []
    for d in range(D + 1):
        rows = monomials(d)
        cols = {m: i for i, m in enumerate(monomials(d + 1))}
        nc = len(cols)
        M = flint.nmod_mat(len(rows), 4 * nc, P)
        for ri, m in enumerate(rows):
            x = {m: R.one}
            for gi, g in enumerate(gens):
                cm = A.comm(x, g)
                for mon, v in cm.items():
                    M[ri, gi * nc + cols[mon]] = int(v)
        dims.append(len(rows) - M.rank())
    return dims


def S_counts(N, r, s, D):
    P = hilbert_numerator(N, r, s)
    return series_from_numerator(P, N, D)


def check_lifts_and_dims(rep, cases, D_of):
    for (N, r, s) in cases:
        tag = f"N={N},r={r},s={s}"
        st = Setup(N, r, s)
        A = st.alg
        hb = hilbert_basis(N, r, s)
        ok_c, ok_lf = True, True
        for m in hb:
            L = st.Lambda(m)
            if not A.is_central(L):
                ok_c = False
            lf = A.leading_form(L)
            if list(lf.keys()) != [m]:
                ok_lf = False
            if A.total_degrees(L) != {sum(m)}:
                ok_lf = False
        rep.check("lem:Lambda Lambda_m central (m in Hilbert basis of S)", tag + f" |HB|={len(hb)}", ok_c)
        rep.check("lem:Lambda lf(Lambda_m) = c x^m, Lambda_m homogeneous", tag, ok_lf)
        D = D_of(N)
        dims = center_dims_modP(N, r, s, D)
        cnt = S_counts(N, r, s, D)
        rep.check("thm:gencenter (i) dim Z(G)_d = #S_d (total degree d <= D)", tag + f", D={D}",
                  dims == cnt, f"Z: {dims} S: {cnt}")
        print(f"    {tag}: dim Z(G)_d, d=0..{D}: {dims}")


def canonical_pairs(N):
    units = [u for u in range(1, N) if gcd(u, N) == 1]
    seen = set()
    for r in range(1, N):
        for s in range(N):
            if not admissible(N, r, s):
                continue
            key = min(((u * r) % N, (u * s) % N) for u in units)
            if key not in seen:
                seen.add(key)
                yield key


def in_minus1_p(N, r, s):
    """h = zeta^s in <-1, p>, p = zeta^r."""
    sub = {(u * r) % N for u in range(N)}
    if s % N in sub:
        return True
    if N % 2 == 0 and (s - N // 2) % N in sub:
        return True
    return False


def check_gorenstein(rep, Nmax=18):
    for N in range(3, Nmax + 1):
        for (r, s) in canonical_pairs(N):
            tag = f"N={N},r={r},s={s}"
            n, ell = order(r, N), order(r + s, N)
            d0 = gcd(n, ell)
            g0 = d0 // gcd(d0, 2)
            pal = is_palindromic(hilbert_numerator(N, r, s))
            g0vec = in_L(N, r, s, (2 * g0, g0, g0, g0))
            crit = in_minus1_p(N, r, s)
            rep.check("thm:gencenter (iii) k[S] Gorenstein (Stanley) <=> h in <-1,p>", tag, pal == crit)
            rep.check("thm:gencenter (iii) proof: (g0,g0,g0,g0) in L_gr <=> h in <-1,p>", tag, g0vec == crit)
            rep.check("lem:arith q^{2g0} = 1 <=> h in <-1,p>", tag, ((2 * g0 * (r + s)) % N == 0) == crit)
            B = L_gr_basis(N, r, s)
            projs = [0, 0, 0, 0]
            for v in B:
                for c in range(4):
                    projs[c] = gcd(projs[c], v[c])
            rep.check("lem:proj pi_c(L_gr) = g0 Z", tag, all(x == g0 for x in projs))


def check_frel(rep, cases):
    for (N, r, s) in cases:
        st = Setup(N, r, s)
        A, R = st.alg, st.R
        e = st.e
        tag = f"N={N},r={r},s={s}"
        As = [st.x1, st.Q(e), st.C, A.power(st.y1, e), A.power(st.y2, e)]
        chis = [st.chi_exp("x1"), st.chi_exp("Q", e), st.chi_exp("C"), st.chi_exp("y1", e), st.chi_exp("y2", e)]
        ok = True
        for i in range(5):
            for j in range(5):
                # lambda_ij from chi_i on the PBW-multidegree of A_j (uniform, see notes)
                lams = set()
                for mono in As[j]:
                    lams.add(sum(c * x for c, x in zip(chis[i], mono)) % N)
                if len(lams) != 1:
                    ok = False
                    continue
                lam = R.zeta(lams.pop())
                if not A.is_zero(A.sub(A.mul(As[i], As[j]), A.scale(lam, A.mul(As[j], As[i])))):
                    ok = False
        rep.check("lem:Ai (i) A_i A_j = lambda_ij A_j A_i", tag, ok)
        A1, A2, A3, A4, A5 = As
        q, p, eps = st.q, st.p, st.eps
        epse, epse1, be, ce = R.one, R.one, R.one, R.one
        for _ in range(e):
            epse = R.red(epse * eps)
            be, ce = R.red(be * st.b), R.red(ce * st.c)
        for _ in range(e - 1):
            epse1 = R.red(epse1 * eps)
        lhs = A.power(A3, e)
        inner = A.sub(A.scale(R.zeta(r * (e * (e - 1) // 2)), A.mul_many(A2, A4, A5)),
                      A.mul(A.power(A1, e), A.add(A.scale(R.red(epse * be), A.mul(A4, A4)),
                                                  A.scale(R.red(epse1 * ce), A.mul(A5, A5)))))
        rhs = A.scale(R.zeta((r + s) * e * (e - 1)), inner)
        rep.check("lem:Ai (ii) eq:Frel", tag, A.is_zero(A.sub(lhs, rhs)))
        # thm:qhyp (i): A^w with w3 < e have distinct leading monomials (checked on a box)
        seen, okb = set(), True
        for w in product(range(3), range(3), range(e), range(3), range(3)):
            m = (w[0], e * w[1] + w[2], e * w[3] + w[2], e * w[4] + w[2])
            if m in seen:
                okb = False
            seen.add(m)
        rep.check("thm:qhyp (i) leading forms of A^w (w3<e) are distinct", tag, okb)


def main():
    rep = Report("06 the center in the PI case (sec:general)")
    small = [(N, r, s) for N in range(3, 7) for r in range(1, N) for s in range(N) if admissible(N, r, s)]
    extra = [(8, 2, 1),   # p of order 4, h of order 8: non-Gorenstein example after thm:gencenter
             (8, 1, 7), (8, 1, 0), (8, 1, 1), (8, 1, 3), (7, 1, 6), (7, 1, 0), (7, 1, 1),
             (10, 1, 4), (12, 1, 11), (12, 1, 0), (12, 1, 1), (12, 1, 5), (9, 1, 8)]
    D_of = lambda N: min(2 * N + 2, 14)
    check_lifts_and_dims(rep, small + extra, D_of)
    # the non-Gorenstein example is indeed not Gorenstein
    rep.check("after thm:gencenter: p of order 4, h of order 8 gives non-Gorenstein center",
              "N=8,r=2,s=1", not is_palindromic(hilbert_numerator(8, 2, 1)))
    check_gorenstein(rep)
    check_frel(rep, small + [(8, 2, 1), (8, 1, 7), (8, 1, 3), (7, 1, 0)])
    finish(rep)


if __name__ == "__main__":
    main()
