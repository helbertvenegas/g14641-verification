"""The helpers behind explore.ipynb give what the paper says.

  ozone structure (invariant factors)     vs cor:ozcases, n <= 40
  center generators are central           all admissible N <= 8, plus the four cases n <= 12
  Hilbert series over generator degrees   hypersurface cases of sec:pres: numerator 1 - t^d
  summary table                           PI degree, Calabi-Yau, Gorenstein vs the lattice checks
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.explore import (center_generators, from_case, hilbert_series, invariant_factors,
                            summary)
from g14641.lattice import (abelian_order_statistics, admissible, ozone_group, pi_degree_formula)
from g14641.report import Report, finish


def expected_iso(n, case):
    if case == "p^-1":
        return [1] if n % 2 else [2, 2]
    if case == "1":
        return [n, n] if n % 4 else [n // 2, 2, n]
    if case == "p":
        return [n, n] if n % 4 else [n // 4, 2, n // 2]
    return [2, 2]


def main():
    rep = Report("09 notebook helpers (explore.py)")
    for n in range(3, 41):
        for case in ("p^-1", "1", "p", "-p^-1"):
            N, r, s = from_case(n, case)
            inv = invariant_factors(ozone_group(N, r, s), N)
            rep.check("explore: Oz(G) invariant factors = cor:ozcases", f"n={n},h={case}",
                      abelian_order_statistics(inv or [1]) == abelian_order_statistics(expected_iso(n, case)))

    cases = [(N, r, s) for N in range(3, 9) for r in range(1, N) for s in range(N) if admissible(N, r, s)]
    cases += [from_case(n, c) for n in range(3, 13) for c in ("p^-1", "1", "p", "-p^-1")]
    for (N, r, s) in cases:
        _, rows = center_generators(N, r, s, check=True)
        rep.check("explore: listed generators are central with the right leading form", f"N={N},r={r},s={s}",
                  all(x["central"] and x["leading_ok"] for x in rows))
        S = summary(N, r, s)
        rep.check("explore: summary PI degree", f"N={N},r={r},s={s}", S["PI degree"] == pi_degree_formula(N, r, s))

    t = sp.symbols("t")
    for n in range(3, 11):
        _, K, degs, pal = hilbert_series(*from_case(n, "p^-1"))
        rep.check("explore: h=p^-1 series numerator is 1 - t^{3n} (hypersurface, thm:center)", f"n={n}",
                  sp.expand(K - (1 - t ** (3 * n))) == 0 and pal)
        if n % 2 == 0:
            _, K, degs, pal = hilbert_series(*from_case(n, "1"))
            rep.check("explore: h=1, n even, numerator 1 - t^{3n} (thm:untwisted)", f"n={n}",
                      sp.expand(K - (1 - t ** (3 * n))) == 0)
        else:
            _, K, degs, pal = hilbert_series(*from_case(n, "1"))
            rep.check("explore: h=1, n odd, polynomial center", f"n={n}", sp.expand(K - 1) == 0 and degs == [n] * 4)
    finish(rep)


if __name__ == "__main__":
    main()
