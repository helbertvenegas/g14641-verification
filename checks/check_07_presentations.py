"""Section sec:pres: presentations for h = p^-1, 1, p (p of order n).

For each case: the listed generators are central (exact, f generic), their
leading exponents generate S (Hilbert basis), the stated relations hold in G
(exact), and the Hilbert series of the stated presentation equals that of
k[S] (which is the Hilbert series of Z(G) by thm:gencenter).
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.lattice import hilbert_basis, hilbert_numerator, series_from_numerator
from g14641.report import Report, finish
from g14641.setup import Setup


def series_rational(num_terms, den_degs, upto):
    """Coefficients of (sum_{(c,d)} c t^d) / prod (1 - t^a) up to degree upto."""
    out = [0] * (upto + 1)
    for c, d in num_terms:
        if d <= upto:
            out[d] += c
    for a in den_degs:
        for i in range(a, upto + 1):
            out[i] += out[i - a]
    return out


def kS_series(N, r, s, upto):
    return series_from_numerator(hilbert_numerator(N, r, s), N, upto)


def rpow(R, x, k):
    out = R.one
    for _ in range(k):
        out = R.red(out * x)
    return out


def leading_exps(A, elts):
    out = []
    for z in elts:
        lf = A.leading_form(z)
        if len(lf) != 1:
            return None
        out.append(next(iter(lf)))
    return sorted(out)


# ------------------------------------------------------------------- h = p^-1
def case_twisted(rep, n):
    st = Setup(n, 1, n - 1)
    A, R = st.alg, st.R
    tag = f"n={n},h=p^-1"
    x1, C, U, Y1, Y2 = st.x1, st.C, st.U(), st.Y(1), st.Y(2)
    gens = [x1, C, U, Y1, Y2]
    rep.check("thm:center (i) x1, C, U, Y1, Y2 central", tag, all(A.is_central(z) for z in gens))
    hb = sorted(hilbert_basis(n, 1, n - 1))
    rep.check("thm:center (i) leading exponents = Hilbert basis of S (eq:centralmon)", tag,
              leading_exps(A, gens) == hb)
    bn, cn = rpow(R, st.b, n), rpow(R, st.c, n)
    x1n = A.power(x1, n)
    base = A.add(A.power(C, n), A.mul(x1n, A.add(A.scale(bn, A.mul(Y1, Y1)), A.scale(cn, A.mul(Y2, Y2)))))
    if n % 2:
        F = A.sub(base, A.mul_many(U, Y1, Y2))
    else:
        m = n // 2
        sgn = R.const((-1) ** m)
        bm, cm = rpow(R, st.b, m), rpow(R, st.c, m)
        corr = A.scale(R.red(2 * sgn), A.mul_many(A.power(x1, m), A.power(C, m),
                                                  A.sub(A.scale(bm, Y1), A.scale(cm, Y2))))
        F = A.add(base, A.mul_many(U, Y1, Y2), corr)
        # L = eps_m Q_m y1^m y2^m = C^m + s x1^m (b^m Y1 - c^m Y2)
        epsm = R.zeta(m * (m - 1) // 2)
        L = A.scale(epsm, A.mul_many(st.Q(m), A.power(st.y1, m), A.power(st.y2, m)))
        rhs = A.add(A.power(C, m), A.scale(sgn, A.mul(A.power(x1, m), A.sub(A.scale(bm, Y1), A.scale(cm, Y2)))))
        rep.check("after eq:LW: L = eps_m Q_m y1^m y2^m = C^m + s x1^m(b^m Y1 - c^m Y2)", tag,
                  A.is_zero(A.sub(L, rhs)) and not A.is_central(st.Q(m)))
    rep.check("prop:relation F_n(x1,Y1,Y2,C,U) = 0", tag, A.is_zero(F))
    upto = 6 * n
    pres = series_rational([(1, 0), (-1, 3 * n)], [1, n, n, 3, n], upto)
    rep.check("thm:center (i) Hilbert series of k[x,Y,Z,C,U]/(F_n) = that of k[S]", tag,
              pres == kS_series(n, 1, n - 1, upto))


def symbolic_twisted(rep):
    X, Y, Z, Cs, Us, B, Cc = sp.symbols("x Y Z C U B Cc")
    for m in range(2, 7):
        s = (-1) ** m
        F = Cs ** (2 * m) + Us * Y * Z + X ** (2 * m) * (B ** 2 * Y ** 2 + Cc ** 2 * Z ** 2) \
            + 2 * s * X ** m * Cs ** m * (B * Y - Cc * Z)
        Lm = Cs ** m + s * X ** m * (B * Y - Cc * Z)
        W = Us + 2 * B * Cc * X ** (2 * m)
        rep.check("eq:LW F_{2m} = L^2 + Y Z W", f"m={m}", sp.expand(F - (Lm ** 2 + Y * Z * W)) == 0)
    for n in range(3, 13):
        epsn = 1 if n % 2 else -1
        if n % 2:
            F = Cs ** n - Us * Y * Z + X ** n * (B * Y ** 2 + Cc * Z ** 2)
        else:
            m = n // 2
            F = Cs ** n + Us * Y * Z + X ** n * (B * Y ** 2 + Cc * Z ** 2) + 2 * (-1) ** m * X ** m * Cs ** m * (Y - Z)
        rep.check("thm:center (iii) F_n|_{x=0} = C^n - eps_n U Y Z", f"n={n}",
                  sp.expand(F.subs(X, 0) - (Cs ** n - epsn * Us * Y * Z)) == 0)


def singular_locus(rep, nlist=(3, 4)):
    """thm:sing for small n with random nonzero values of the parameters:
    every component lies in V(F, dF) and V(F, dF) lies in the union."""
    x, Y, Z, Cs, U, w = sp.symbols("x Y Z C U w")
    V = [x, Y, Z, Cs, U]
    for n in nlist:
        tag = f"n={n}"
        if n % 2:
            bn, cn = sp.Rational(3, 2), sp.Rational(-5, 7)
            F = Cs ** n - U * Y * Z + x ** n * (bn * Y ** 2 + cn * Z ** 2)
            comps = [[Y, Z, Cs], [x, U, Y, Cs], [x, U, Z, Cs]]
        else:
            m = n // 2
            s = (-1) ** m
            bm, cm = sp.Rational(3, 2), sp.Rational(-5, 7)
            F = Cs ** n + U * Y * Z + x ** n * (bm ** 2 * Y ** 2 + cm ** 2 * Z ** 2) \
                + 2 * s * x ** m * Cs ** m * (bm * Y - cm * Z)
            comps = [[Y, Z, Cs], [Y, U + 2 * bm * cm * x ** n, Cs ** m - s * cm * x ** m * Z],
                     [Z, U + 2 * bm * cm * x ** n, Cs ** m + s * bm * x ** m * Y]]
        J = [F] + [sp.diff(F, v) for v in V]
        ok_in = True
        for I in comps:
            GB = sp.groebner(I, *V, order="grevlex", domain=sp.QQ)
            if not all(GB.reduce(sp.expand(g))[1] == 0 for g in J):
                ok_in = False
        rep.check("thm:sing each component lies in Sing", tag, ok_in)
        ok_cov = True
        for a in comps[0]:
            for b in comps[1]:
                for c in comps[2]:
                    g = sp.expand(a * b * c)
                    GB = sp.groebner(J + [1 - w * g], w, *V, order="grevlex", domain=sp.QQ)
                    if not (len(GB.exprs) == 1 and GB.exprs[0] == 1):
                        ok_cov = False
        rep.check("thm:sing Sing lies in the union of the components", tag, ok_cov)


# ------------------------------------------------------------------- h = 1
def case_untwisted(rep, n):
    st = Setup(n, 1, 0)
    A, R = st.alg, st.R
    tag = f"n={n},h=1"
    x1n, Y1, Y2, U = A.power(st.x1, n), st.Y(1), st.Y(2), st.U()
    if n % 2:
        gens = [x1n, Y1, Y2, U]
        rep.check("thm:untwisted (i) generators central", tag, all(A.is_central(z) for z in gens))
        rep.check("thm:untwisted (i) S = n N^4 (polynomial center)", tag,
                  leading_exps(A, gens) == sorted(hilbert_basis(n, 1, 0)))
        upto = 5 * n
        rep.check("thm:untwisted (i) Hilbert series 1/(1-t^n)^4", tag,
                  series_rational([(1, 0)], [n] * 4, upto) == kS_series(n, 1, 0, upto))
    else:
        m = n // 2
        P = st.Q(m)
        Ups = A.mul_many(A.power(st.x1, m), A.power(st.y1, m), A.power(st.y2, m))
        gens = [x1n, Y1, Y2, P, Ups]
        rep.check("thm:untwisted (ii) generators central", tag, all(A.is_central(z) for z in gens))
        rep.check("thm:untwisted (ii) leading exponents = Hilbert basis of S", tag,
                  leading_exps(A, gens) == sorted(hilbert_basis(n, 1, 0)))
        rel = A.sub(A.mul(Ups, Ups), A.scale(R.const((-1) ** m), A.mul_many(x1n, Y1, Y2)))
        rep.check("thm:untwisted (ii) eq:untwrel", tag, A.is_zero(rel))
        upto = 6 * n
        pres = series_rational([(1, 0), (-1, 6 * m)], [n, n, n, m, 3 * m], upto)
        rep.check("thm:untwisted (ii) Hilbert series of the presentation = k[S]", tag,
                  pres == kS_series(n, 1, 0, upto))
    ser = kS_series(n, 1, 0, n)
    rep.check("rem:rubiano degree-n component of Z(G) has dimension 4", tag, ser[n] == 4 + (1 if n == 2 else 0))


# ------------------------------------------------------------------- h = p
def case_hp(rep, n):
    st = Setup(n, 1, 1)
    A, R = st.alg, st.R
    tag = f"n={n},h=p"
    Y1, Y2, U = st.Y(1), st.Y(2), st.U()
    if n % 2:
        gens = [A.power(st.x1, n), Y1, Y2, U]
        rep.check("thm:hpcenter (i) generators central", tag, all(A.is_central(z) for z in gens))
        rep.check("thm:hpcenter (i) S = n N^4 (polynomial center)", tag,
                  leading_exps(A, gens) == sorted(hilbert_basis(n, 1, 1)))
    elif (n // 2) % 2:
        m = n // 2
        X = A.power(st.x1, m)
        W = A.mul_many(st.Q(m), A.power(st.y1, m), A.power(st.y2, m))
        gens = [X, Y1, Y2, U, W]
        rep.check("thm:hpcenter (ii) generators central", tag, all(A.is_central(z) for z in gens))
        rep.check("thm:hpcenter (ii) leading exponents = Hilbert basis of S", tag,
                  leading_exps(A, gens) == sorted(hilbert_basis(n, 1, 1)))
        am = rpow(R, st.alpha, m)
        rel = A.add(A.mul(W, W), A.mul_many(Y1, Y2, A.add(U, A.scale(R.red(2 * am), A.mul(X, X)))))
        rep.check("thm:hpcenter (ii) eq:hprel", tag, A.is_zero(rel))
        upto = 6 * n
        pres = series_rational([(1, 0), (-1, 6 * m)], [m, n, n, n, 3 * m], upto)
        rep.check("thm:hpcenter (ii) Hilbert series of the presentation = k[S]", tag,
                  pres == kS_series(n, 1, 1, upto))
    else:
        k = n // 4
        m = 2 * k
        x1, y1, y2 = st.x1, st.y1, st.y2
        v1, v2, v3 = A.power(x1, k), st.Q(2 * k), A.power(st.C, k)
        v4 = A.mul(A.power(y1, 2 * k), A.power(y2, 2 * k))
        Ck = v3
        ten = [A.power(x1, 2 * k), A.mul(v1, v2), Y1, Y2, U, A.mul(v1, Ck), A.mul(v2, Ck),
               A.mul(v1, v4), A.mul(v2, v4), A.mul(Ck, v4)]
        rep.check("thm:hpcenter (iii) the ten elements are central", tag, all(A.is_central(z) for z in ten))
        hb = sorted(hilbert_basis(n, 1, 1))
        rep.check("thm:hpcenter (iii) leading exponents = Hilbert basis of S (10 elements)", tag,
                  leading_exps(A, ten) == hb and len(hb) == 10)
        vs = [v1, v2, v3, v4]
        ok = True
        for i in range(4):
            for j in range(4):
                ok &= A.is_zero(A.comm(vs[i], vs[j]))
            ok &= A.is_zero(A.comm(vs[i], x1)) and A.is_zero(A.comm(vs[i], st.x2))
            ok &= A.is_zero(A.add(A.mul(vs[i], y1), A.mul(y1, vs[i])))
            ok &= A.is_zero(A.add(A.mul(vs[i], y2), A.mul(y2, vs[i])))
        rep.check("prop:hppres (i) v_i commute, commute with x1,x2, anticommute with y1,y2", tag, ok)
        eps = R.zeta(k * (2 * k - 1))
        bm, cm = rpow(R, st.b, m), rpow(R, st.c, m)
        r1 = A.sub(A.mul(v4, v4), A.mul(Y1, Y2))
        r2 = A.sub(A.mul(v3, v3), A.sub(A.scale(eps, A.mul(v2, v4)),
                                        A.mul(A.mul(v1, v1), A.sub(A.scale(bm, Y1), A.scale(cm, Y2)))))
        rep.check("prop:hppres (i) eq:vrel", tag, A.is_zero(r1) and A.is_zero(r2))
        am = rpow(R, st.alpha, m)
        rep.check("prop:hppres (iii) U = v2^2 - 2 alpha^m v1^4", tag,
                  A.is_zero(A.sub(U, A.sub(A.mul(v2, v2), A.scale(R.red(2 * am), A.power(v1, 4))))))
        upto = 6 * n
        basis_degs = [0, 3 * k, 4 * k, 5 * k, 5 * k, 6 * k, 7 * k, 10 * k]
        pres = series_rational([(1, d) for d in basis_degs], [2 * k, n, n, n], upto)
        rep.check("prop:hppres (iv) free of rank 8 over R0: Hilbert series = k[S]", tag,
                  pres == kS_series(n, 1, 1, upto))


def main():
    rep = Report("07 presentations for h = p^-1, 1, p (sec:pres)")
    symbolic_twisted(rep)
    for n in range(3, 11):
        case_twisted(rep, n)
        case_untwisted(rep, n)
    for n in range(3, 13):
        case_hp(rep, n)
    singular_locus(rep)
    finish(rep)


if __name__ == "__main__":
    main()
