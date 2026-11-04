"""lem:index, thm:PIdeg and the special values in Theorem A(ii).

[Z^4 : L] is computed from the Smith normal form of Omega (no use of the
Pfaffian) and, for N <= 10, also by listing the image of Omega on (Z/N)^4.
Both are compared with (N^2 / gcd(N, 2r(r+s)))^2.
"""

import os
import sys
from math import gcd, isqrt

import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.lattice import (admissible, image_size_bruteforce, lattice_L, omega_paper,
                            pi_degree_formula)
from g14641.report import Report, finish


def main(Nmax=60):
    rep = Report("03 PI degree (lem:index, thm:PIdeg)")

    r_, s_ = sp.symbols("r s", integer=True)
    O = omega_paper(r_, s_)
    pf = sp.expand(O[0][1] * O[2][3] - O[0][2] * O[1][3] + O[0][3] * O[1][2])
    rep.check("lem:index Pf(Omega) = 2r(r+s)", "symbolic", sp.expand(pf - 2 * r_ * (r_ + s_)) == 0)

    count = 0
    for N in range(3, Nmax + 1):
        for r in range(1, N):
            for s in range(N):
                if not admissible(N, r, s):
                    continue
                count += 1
                case = f"N={N},r={r},s={s}"
                _, idx = lattice_L(N, r, s)
                pd = pi_degree_formula(N, r, s)
                rep.check("lem:index [Z^4:L] = (N^2/gcd(N,2r(r+s)))^2 (Smith form)", case, idx == pd * pd,
                          f"index={idx}, formula^2={pd * pd}")
                rep.check("thm:PIdeg PIdeg = sqrt[Z^4:L] is an integer", case, isqrt(idx) ** 2 == idx)
                if N <= 10:
                    rep.check("lem:index [Z^4:L] = |image of Omega mod N| (brute force)", case,
                              image_size_bruteforce(N, r, s) == idx)

    # special values: zeta = p, N = n, r = 1
    for n in range(3, Nmax + 1):
        for s, expect, name in [(0, n * n // gcd(n, 2), "h=1"), (1, n * n // gcd(n, 4), "h=p"), (-1, n, "h=p^-1")]:
            _, idx = lattice_L(n, 1, s % n)
            rep.check(f"thm:PIdeg special value {name}", f"n={n}", isqrt(idx) == expect and isqrt(idx) ** 2 == idx)
        # Zhang twist: h = p^-1 vs h = 1 for n odd (Introduction)
        if n % 2:
            rep.check("intro: twist h=p^-1 -> h=1 changes PIdeg n -> n^2", f"n={n}",
                      isqrt(lattice_L(n, 1, n - 1)[1]) == n and isqrt(lattice_L(n, 1, 0)[1]) == n * n)

    print(f"  ({count} admissible triples with N <= {Nmax})")
    finish(rep)


if __name__ == "__main__":
    main()
