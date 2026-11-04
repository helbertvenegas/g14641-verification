"""Helpers for the interactive notebook (explore.ipynb).

Everything here is built on the same code as the checks: the lattice L,
the semigroup S, the central lifts Lambda_m and the exact multiplication in G.
"""

from math import gcd

import sympy as sp

from .lattice import (admissible, hilbert_basis, hilbert_numerator, in_L, is_palindromic,
                      lattice_L, order, ozone_group, pi_degree_formula, series_from_numerator)
from .setup import Setup

GEN_NAMES = ("x_1", "x_2", "y_1", "y_2")


# --------------------------------------------------------------- parameters
def from_case(n, h):
    """p a primitive n-th root of unity and h in {'p^-1', '1', 'p', '-p^-1'}.
    Returns (N, r, s) with zeta of order N, p = zeta^r, h = zeta^s."""
    if n < 3:
        raise ValueError("the paper assumes n = ord(p) >= 3 in these cases")
    if h == "p^-1":
        return n, 1, n - 1
    if h == "1":
        return n, 1, 0
    if h == "p":
        return n, 1, 1
    if h == "-p^-1":
        return (n, 1, n // 2 - 1) if n % 2 == 0 else (2 * n, 2, n - 2)
    raise ValueError("h must be one of 'p^-1', '1', 'p', '-p^-1'")


def check_params(N, r, s):
    r, s = r % N, s % N
    if gcd(gcd(r, s), N) != 1:
        raise ValueError(f"gcd(r, s, N) = {gcd(gcd(r, s), N)} != 1: then zeta^r, zeta^s do not "
                         f"generate mu_N. Divide N, r, s by the gcd.")
    if (2 * r) % N == 0:
        raise ValueError("p^2 = 1 is excluded in the paper.")
    return N, r, s


def in_minus1_p(N, r, s):
    sub = {(u * r) % N for u in range(N)}
    return (s % N in sub) or (N % 2 == 0 and (s - N // 2) % N in sub)


def root(e, N):
    e %= N
    if e == 0:
        return "1"
    g = gcd(e, N)
    if N // g == 2:
        return "-1"
    return rf"\zeta^{{{e}}}" if e != 1 else r"\zeta"


# ------------------------------------------------------------- ozone groups
def invariant_factors(G, N):
    """Invariant factors of a finite abelian group given as a list of
    (a, t, c) in Z/N x Z/2 x Z/N."""
    def tors(d):
        return sum(1 for (a, t, c) in G if (d * a) % N == 0 and (d * t) % 2 == 0 and (d * c) % N == 0)

    def vlog(m, p):
        k = 0
        while m % p == 0:
            m //= p
            k += 1
        assert m == 1
        return k

    cyclic = []
    for p in sp.primefactors(len(G)):
        sizes = [0]                      # sizes[k] = log_p |G[p^k]|
        k = 1
        while True:
            sizes.append(vlog(tors(p ** k), p))
            if sizes[k] == sizes[k - 1]:
                break
            k += 1
        # a[k] = number of cyclic p-factors of order >= p^k
        a = [sizes[k] - sizes[k - 1] for k in range(1, len(sizes))] + [0]
        parts = []
        for k in range(len(a) - 1):
            parts += [p ** (k + 1)] * (a[k] - a[k + 1])
        cyclic.append(sorted(parts, reverse=True))
    width = max((len(c) for c in cyclic), default=0)
    inv = []
    for i in range(width):
        f = 1
        for c in cyclic:
            if i < len(c):
                f *= c[i]
        inv.append(f)
    return sorted(inv)


def ozone_latex(N, r, s, list_max=16):
    G = ozone_group(N, r, s)
    inv = invariant_factors(G, N)
    struct = r" \times ".join(rf"\mu_{{{d}}}" for d in inv) if inv else r"\{1\}"
    elems = []
    if len(G) <= list_max:
        for (a, t, c) in sorted(G):
            elems.append(rf"\varphi_{{{root(a, N)},\,{'-1' if t else '1'},\,{root(c, N)}}}")
    return G, struct, elems


# ------------------------------------------------------------------ center
def lift_formula(m, e):
    m1, m2, m3, m4 = m
    j0 = m2 % e
    sig = (m2 - j0) // e
    parts = []

    def pw(sym, k):
        if k == 0:
            return None
        return sym if k == 1 else f"{sym}^{{{k}}}"

    for sym, k in ((f"Q_{{{e}}}", sig), ("x_1", m1), ("C", j0), ("y_1", m3 - j0), ("y_2", m4 - j0)):
        t = pw(sym, k)
        if t:
            parts.append(t)
    return " ".join(parts) if parts else "1"


def monomial_latex(m):
    out = []
    for g, k in zip(GEN_NAMES, m):
        if k == 1:
            out.append(g)
        elif k > 1:
            out.append(f"{g}^{{{k}}}")
    return " ".join(out) if out else "1"


def center_generators(N, r, s, check=True):
    """Generators Lambda_m of Z(G), m in the Hilbert basis of S (thm:gencenter (i)).
    With check=True each one is tested for centrality by exact multiplication
    in G, with f an indeterminate."""
    N, r, s = check_params(N, r, s)
    st = Setup(N, r, s)
    rows = []
    for m in sorted(hilbert_basis(N, r, s), key=lambda v: (sum(v), v)):
        row = {"m": m, "degree": sum(m), "lift": lift_formula(m, st.e),
               "leading": monomial_latex(m)}
        if check:
            L = st.Lambda(m)
            row["central"] = st.alg.is_central(L)
            row["terms"] = len(L)
            row["leading_ok"] = list(st.alg.leading_form(L).keys()) == [m]
        rows.append(row)
    return st, rows


def hilbert_series(N, r, s, upto=None):
    N, r, s = check_params(N, r, s)
    P = hilbert_numerator(N, r, s)
    upto = 3 * N if upto is None else upto
    coeffs = series_from_numerator(P, N, upto)
    t = sp.symbols("t")
    num = sum(c * t ** i for i, c in enumerate(P) if c)
    H = num / (1 - t ** N) ** 4
    # write H over prod (1 - t^d), d the degrees of the generators Lambda_m
    degs = sorted(sum(m) for m in hilbert_basis(N, r, s))
    den = sp.prod([(1 - t ** d) for d in degs])
    K = sp.expand(sp.cancel(H * den))
    assert K.is_polynomial(t)
    return coeffs, K, degs, is_palindromic(P)


def series_latex(K, degs):
    t = sp.symbols("t")
    from collections import Counter
    def fac(d, k):
        base = "(1-t)" if d == 1 else rf"(1-t^{{{d}}})"
        return base if k == 1 else rf"{base}^{{{k}}}"
    den = " ".join(fac(d, k) for d, k in sorted(Counter(degs).items()))
    num = sp.latex(sp.Poly(K, t).as_expr(), order="old")
    return rf"H_{{Z(G)}}(t) = \frac{{{num}}}{{{den}}}"


# ----------------------------------------------------------------- summary
def summary(N, r, s):
    N, r, s = check_params(N, r, s)
    n, e, ell = order(r, N), order(2 * r, N), order(r + s, N)
    _, idx = lattice_L(N, r, s)
    pdeg = pi_degree_formula(N, r, s)
    assert pdeg * pdeg == idx
    cy = (2 * (r + s)) % N == 0
    gor = in_minus1_p(N, r, s)
    G = ozone_group(N, r, s)
    hdet_triv = all((2 * a + 2 * c) % N == 0 for (a, t, c) in G)
    hb = hilbert_basis(N, r, s)
    return {
        "$N$, $r$, $s$": f"{N}, {r}, {s}",
        r"$n=\operatorname{ord}(p)$": n,
        r"$e=\operatorname{ord}(p^2)$": e,
        r"$\operatorname{ord}(q)$, $q=hp$": ell,
        "PI degree": pdeg,
        r"rank of $G$ over $Z(G)$": idx,
        r"Calabi–Yau ($h=\pm p^{-1}$)": cy,
        "Nakayama automorphism": rf"$\nu_G = \varphi_{{{root(-2 * (r + s), N)},\,1,\,{root(2 * (r + s), N)}}}$",
        r"$Z(G)$ Gorenstein ($h\in\langle -1,p\rangle$)": gor,
        r"order of $\operatorname{Oz}(G)$": len(G),
        r"$\operatorname{hdet}=1$ on $\operatorname{Oz}(G)$": hdet_triv,
        r"generators of $Z(G)$ (Hilbert basis of $S$)": len(hb),
        r"$k[S]$ is a polynomial ring": len(hb) == 4,
    }


# ------------------------------------------------------------- printing
def coeff_str(c):
    s = str(c)
    return s.replace("z", "ζ").replace("*", "·")


def element_str(elt, max_terms=40):
    items = sorted(elt.items(), key=lambda kv: (-sum(kv[0]), kv[0]))
    parts = []
    for m, c in items[:max_terms]:
        mon = " ".join(f"{g.replace('_', '')}^{k}" if k > 1 else g.replace("_", "")
                       for g, k in zip(GEN_NAMES, m) if k) or "1"
        parts.append(f"({coeff_str(c)})·{mon}")
    more = f"  + ... ({len(items) - max_terms} more terms)" if len(items) > max_terms else ""
    return "\n + ".join(parts) + more if parts else "0"
