"""Section sec:prelim: presentation, PBW basis, Ore data, commutation rules.

All checks here are for generic p, h, f (coefficients in Q[p,h,f]), except the
associativity test, which is repeated over Q(zeta_N)[f].
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g14641.algebra import GAlgebra, GENS
from g14641.report import Report, finish
from g14641.rings import CycRing, GenericRing


def main():
    rep = Report("01 presentation (sec:prelim)")
    R = GenericRing()
    A = GAlgebra(R, R.p, R.h)
    p, h, f = R.p, R.h, R.f
    q = h * p
    x1, x2, y1, y2 = (A.gen(g) for g in GENS)
    one = R.one

    # --- associativity on all generator triples and on random elements -----
    for ring_name, alg in [("Q[p,h,f]", A),
                           ("Q(zeta_12)[f], p=z, h=z^5", GAlgebra(CycRing(12), CycRing(12).zeta(1), CycRing(12).zeta(5))),
                           ("Q(zeta_8)[f], p=z^2, h=z", GAlgebra(CycRing(8), CycRing(8).zeta(2), CycRing(8).zeta(1)))]:
        gens = [alg.gen(g) for g in GENS]
        ok = True
        for a, b, c in itertools.product(gens, repeat=3):
            if not alg.is_zero(alg.sub(alg.mul(alg.mul(a, b), c), alg.mul(a, alg.mul(b, c)))):
                ok = False
        rep.check("prop:ore PBW basis: associativity on generator triples", ring_name, ok)
        random.seed(7)
        ok = True
        for _ in range(12):
            el = []
            for d in (3, 2, 3):
                x = {}
                for _ in range(4):
                    m = [0, 0, 0, 0]
                    for _ in range(d):
                        m[random.randrange(4)] += 1
                    x[tuple(m)] = alg.R.const(random.randint(1, 5))
                el.append(x)
            a, b, c = el
            if not alg.is_zero(alg.sub(alg.mul(alg.mul(a, b), c), alg.mul(a, alg.mul(b, c)))):
                ok = False
        rep.check("prop:ore PBW basis: associativity on random elements", ring_name, ok)

    # --- relations eq:GZZ of [ZZ2] after x2 = x2~ + (p-1)^{-1} x1 -------------
    # multiply by (p-1): w = (p-1) x2~ = (p-1) x2 - x1
    w = A.sub(A.scale(p - one, x2), x1)
    lhs = A.mul(y1, w)
    rhs = A.scale(h, A.add(A.scale(p * (p - one), A.mul(x1, y1)),
                           A.scale(p * p, A.mul(w, y1)),
                           A.scale(p - one, A.mul(x1, y2))))
    rep.check("eq:GZZ -> eq:G: relation for y1 x2~", "generic", A.is_zero(A.sub(lhs, rhs)))
    lhs = A.mul(y2, w)
    rhs = A.scale(h, A.add(A.scale(f * (p - one), A.mul(x1, y1)),
                           A.scale(-(p - one), A.mul(x1, y2)),
                           A.mul(w, y2)))
    rep.check("eq:GZZ -> eq:G: relation for y2 x2~", "generic", A.is_zero(A.sub(lhs, rhs)))
    rep.check("eq:GZZ -> eq:G: x2~ commutes with x1", "generic", A.is_zero(A.comm(w, x1)))

    # --- prop:ore: delta is a sigma-derivation (three displayed identities) --
    # multiplied by h p^2 to clear sigma(y1) = h^-1 p^-2 y1, sigma(y2) = h^-1 y2
    # sigma(x1) = x1, delta(x1) = 0, delta(y1) = -p^-2 x1 y2, delta(y2) = -f x1 y1
    # identity 1: sigma(y1)d(x1) + d(y1)x1 = hp(sigma(x1)d(y1) + d(x1)y1)
    #   <=> -p^-2 x1 y2 x1 = hp x1 (-p^-2 x1 y2)   (times p^2)
    lhs = A.scale(-one, A.mul_many(x1, y2, x1))
    rhs = A.scale(-q, A.mul_many(x1, x1, y2))
    rep.check("prop:ore delta is a sigma-derivation (y1,x1)", "generic", A.is_zero(A.sub(lhs, rhs)))
    lhs = A.scale(-f, A.mul_many(x1, y1, x1))
    rhs = A.scale(-q * f, A.mul_many(x1, x1, y1))
    rep.check("prop:ore delta is a sigma-derivation (y2,x1)", "generic", A.is_zero(A.sub(lhs, rhs)))
    # identity 3 times h p^2:
    # sigma(y2)d(y1) + d(y2)y1 = p(sigma(y1)d(y2) + d(y1)y2)
    # LHS*hp^2 = y2(-x1 y2) + hp^2(-f x1 y1) y1 ; RHS*hp^2 = p( y1(-f x1 y1) + h(-x1 y2) y2 )
    lhs = A.add(A.scale(-one, A.mul_many(y2, x1, y2)), A.scale(-h * p * p * f, A.mul_many(x1, y1, y1)))
    rhs = A.scale(p, A.add(A.scale(-f, A.mul_many(y1, x1, y1)), A.scale(-h, A.mul_many(x1, y2, y2))))
    rep.check("prop:ore delta is a sigma-derivation (y2,y1)", "generic", A.is_zero(A.sub(lhs, rhs)))
    # Ore commutation x2 r = sigma(r) x2 + delta(r), times h p^2 resp. h
    lhs = A.scale(h * p * p, A.mul(x2, y1))
    rhs = A.sub(A.mul(y1, x2), A.scale(h, A.mul(x1, y2)))
    rep.check("prop:ore eq:sigmadelta x2 y1", "generic", A.is_zero(A.sub(lhs, rhs)))
    lhs = A.scale(h, A.mul(x2, y2))
    rhs = A.sub(A.mul(y2, x2), A.scale(h * f, A.mul(x1, y1)))
    rep.check("prop:ore eq:sigmadelta x2 y2", "generic", A.is_zero(A.sub(lhs, rhs)))

    # --- rem:pbw: skew PBW relations -------------------------------------
    ok = (A.is_zero(A.sub(A.sub(A.mul(y1, x2), A.scale(h * p * p, A.mul(x2, y1))), A.scale(h, A.mul(x1, y2))))
          and A.is_zero(A.sub(A.sub(A.mul(y2, x2), A.scale(h, A.mul(x2, y2))), A.scale(h * f, A.mul(x1, y1))))
          and A.is_zero(A.sub(A.mul(y2, y1), A.scale(p, A.mul(y1, y2)))))
    rep.check("rem:pbw skew PBW relations", "generic", ok)

    # --- lem:rules ------------------------------------------------------------
    for r in range(1, 9):
        y1r, y2r = A.power(y1, r), A.power(y2, r)
        qint = R.zero
        for i in range(r):
            qint = qint + (p * p) ** i
        ok1 = A.is_zero(A.sub(A.mul(y1r, x1), A.scale(q ** r, A.mul(x1, y1r)))) and \
              A.is_zero(A.sub(A.mul(y2r, x1), A.scale(q ** r, A.mul(x1, y2r))))
        ok2 = A.is_zero(A.sub(A.mul(y2, y1r), A.scale(p ** r, A.mul(y1r, y2)))) and \
              A.is_zero(A.sub(A.mul(y2r, y1), A.scale(p ** r, A.mul(y1, y2r))))
        rhs1 = A.add(A.scale((h * p * p) ** r, A.mul(x2, y1r)),
                     A.scale(h ** r * qint, A.mul_many(x1, y2, A.power(y1, r - 1))))
        rhs2 = A.add(A.scale(h ** r, A.mul(x2, y2r)),
                     A.scale(h ** r * f * qint, A.mul_many(x1, y1, A.power(y2, r - 1))))
        ok3 = A.is_zero(A.sub(A.mul(y1r, x2), rhs1)) and A.is_zero(A.sub(A.mul(y2r, x2), rhs2))
        rep.check("lem:rules eq:rules1", f"r={r}", ok1)
        rep.check("lem:rules eq:rules2", f"r={r}", ok2)
        rep.check("lem:rules eq:yrX", f"r={r}", ok3)

    # --- eq:phixi: phi_{xi,theta,eta} preserves the relations ----------------
    # check on the relations with xi, eta as extra indeterminates: we use the
    # cyclotomic ring Q(zeta_N)[f] with xi = zeta^a, eta = zeta^c and all a, c.
    for N, rr, ss in [(5, 1, 3), (6, 1, 0), (8, 2, 1)]:
        Rc = CycRing(N)
        Ac = GAlgebra(Rc, Rc.zeta(rr), Rc.zeta(ss))
        ok = True
        for a in range(N):
            for c in range(N):
                for th in (1, -1):
                    xi, eta, T = Rc.zeta(a), Rc.zeta(c), Rc.const(th)
                    img = {"x1": Ac.scale(xi, Ac.gen("x1")), "x2": Ac.scale(Rc.red(T * xi), Ac.gen("x2")),
                           "y1": Ac.scale(eta, Ac.gen("y1")), "y2": Ac.scale(Rc.red(T * eta), Ac.gen("y2"))}
                    hh, pp = Ac.h, Ac.p
                    rels = [
                        Ac.sub(Ac.mul(img["x2"], img["x1"]), Ac.mul(img["x1"], img["x2"])),
                        Ac.sub(Ac.mul(img["y1"], img["x1"]), Ac.scale(Ac.q, Ac.mul(img["x1"], img["y1"]))),
                        Ac.sub(Ac.mul(img["y2"], img["x1"]), Ac.scale(Ac.q, Ac.mul(img["x1"], img["y2"]))),
                        Ac.sub(Ac.mul(img["y2"], img["y1"]), Ac.scale(pp, Ac.mul(img["y1"], img["y2"]))),
                        Ac.sub(Ac.mul(img["y1"], img["x2"]),
                               Ac.scale(hh, Ac.add(Ac.scale(Rc.red(pp * pp), Ac.mul(img["x2"], img["y1"])),
                                                   Ac.mul(img["x1"], img["y2"])))),
                        Ac.sub(Ac.mul(img["y2"], img["x2"]),
                               Ac.scale(hh, Ac.add(Ac.mul(img["x2"], img["y2"]),
                                                   Ac.scale(Ac.f, Ac.mul(img["x1"], img["y1"]))))),
                    ]
                    if not all(Ac.is_zero(x) for x in rels):
                        ok = False
        rep.check("eq:phixi phi_{xi,theta,eta} preserves eq:G", f"N={N},r={rr},s={ss}, all xi,eta in mu_N", ok)

    # --- independent rewriting engine ----------------------------------------
    # Words in the generators are reduced to PBW normal form by rewriting with
    # the six relations eq:G only (no matrix rule, no caching).  Its products
    # must agree with GAlgebra.mul.
    order_ = {"x1": 0, "x2": 1, "y1": 2, "y2": 3}
    rules = {
        ("x2", "x1"): [(one, ("x1", "x2"))],
        ("y1", "x1"): [(q, ("x1", "y1"))],
        ("y2", "x1"): [(q, ("x1", "y2"))],
        ("y2", "y1"): [(p, ("y1", "y2"))],
        ("y1", "x2"): [(h * p * p, ("x2", "y1")), (h, ("x1", "y2"))],
        ("y2", "x2"): [(h, ("x2", "y2")), (h * f, ("x1", "y1"))],
    }

    def rewrite(word):
        todo = {tuple(word): one}
        done = {}
        while todo:
            w, c = todo.popitem()
            for t in range(len(w) - 1):
                if order_[w[t]] > order_[w[t + 1]]:
                    for cc, pair in rules[(w[t], w[t + 1])]:
                        nw = w[:t] + pair + w[t + 2:]
                        todo[nw] = todo.get(nw, R.zero) + c * cc
                    break
            else:
                key = tuple(w.count(g) for g in GENS)
                done[key] = done.get(key, R.zero) + c
        return {k: v for k, v in done.items() if v != 0}

    random.seed(11)
    ok = True
    for _ in range(60):
        L1, L2 = random.randint(1, 5), random.randint(1, 5)
        w1 = [random.choice(GENS) for _ in range(L1)]
        w2 = [random.choice(GENS) for _ in range(L2)]
        a, b = rewrite(w1), rewrite(w2)
        if not A.is_zero(A.sub(A.mul(a, b), rewrite(w1 + w2))):
            ok = False
    rep.check("engine: GAlgebra.mul agrees with naive rewriting by eq:G", "60 random word pairs, generic", ok)

    # --- negative controls: wrong identities must be detected ----------------
    bad1 = A.sub(A.mul(y1, x2), A.scale(h * p * p, A.mul(x2, y1)))           # drops h x1 y2
    bad2 = A.sub(A.mul(A.power(y1, 3), x2), A.scale((h * p * p) ** 3, A.mul(x2, A.power(y1, 3))))
    rep.check("control: a wrong identity is reported as nonzero", "generic",
              not A.is_zero(bad1) and not A.is_zero(bad2))

    finish(rep)


if __name__ == "__main__":
    main()
