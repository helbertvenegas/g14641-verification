# Computational checks for the family G(p,f,h) of type (14641)

Companion code for

> J. Gómez and H. Venegas, *PI degree, center and ozone group of the double extension family G of type (14641)*.

Every result in the paper is proved by hand. This repository is a second, independent check of the computations, mostly because the center is hard to verify by eye. The central lifts mix Dickson polynomials with the normal element C, and a wrong sign or a wrong power of p there would not be noticed by reading. None of the proofs depends on this code.

## What is checked

The algebra is implemented from its defining relations only, through the PBW basis `x1^i x2^j y1^k y2^l`. Products are computed exactly. In the root-of-unity case the coefficients live in **Q(ζ_N)[f], with f an indeterminate**, so the checks hold for every value of f and not just for one number. For identities that hold for all parameters, the coefficients live in Q[p,h,f]. The multiplication is checked against a naive rewriting with the six relations and against associativity on random elements before anything else is trusted.

Parameters follow Section 3 of the paper: ζ is a primitive N-th root of unity, `p = ζ^r`, `h = ζ^s`, with `gcd(r,s,N) = 1` (so `N = |<p,h>|`) and `p² ≠ 1`. "All admissible" below means all such triples (N, r, s) in the stated range.

| Script | Statements (LaTeX labels) | Range | Arithmetic |
|---|---|---|---|
| `check_01_presentation.py` | `eq:GZZ`→`eq:G`, `prop:ore` (σ-derivation, Ore data), `rem:pbw`, `lem:rules`, `eq:phixi` | generic p, h, f; r ≤ 8 | exact |
| `check_02_birational.py` | `thm:birational` (the map x2 ↦ x1(T+u) respects `eq:G`), `lem:inner`, `eq:t`, `eq:C`, `rem:CGWZ17`, `eq:Omega`, `eq:rhobeta`, `lem:normal`(iv), `prop:rationalcenter`, `prop:torus` | generic p, h, f; all admissible N ≤ 12 | exact |
| `check_03_pi_degree.py` | `lem:index`, `thm:PIdeg`, Theorem A(ii), special values for h = 1, p, p⁻¹, the twist remark in the introduction | all admissible N ≤ 60 (59 398 triples) | exact (Smith normal form; brute force for N ≤ 10) |
| `check_04_normal_elements.py` | `prop:dickson`, `lem:qdickson`, `lem:grmon`, `lem:normal`(ii),(iii), the paragraph after `lem:normal`, `lem:finite` | all admissible N ≤ 10 (`lem:grmon` N ≤ 6, `lem:finite` N ≤ 5) | exact |
| `check_05_ozone.py` | `thm:ozall`, `cor:ozcases` (element by element and isomorphism type), Theorem B, `rem:ozCY`, `lem:hdet`, `thm:hdetCY` | n ≤ 40 for the four cases; all admissible N ≤ 30 for the general statements; the lattice description is tested against the algebra for N ≤ 6 | exact |
| `check_06_center_general.py` | `lem:Lambda`, `thm:gencenter`(i),(iii), `lem:proj`, `lem:arith`, the non-Gorenstein example, `lem:Ai`, `eq:Frel`, `thm:qhyp`(i) | lifts and dimensions: all admissible N ≤ 6 plus 14 cases up to N = 12; Gorenstein: all admissible N ≤ 18, one triple per choice of ζ | exact, except the upper bound in `thm:gencenter`(i) (see below) |
| `check_07_presentations.py` | `prop:relation`, `eq:LW`, `thm:center`, `thm:sing` (n = 3, 4), `thm:untwisted`, `rem:rubiano`, `thm:hpcenter`, `prop:hppres` | n ≤ 10 for h = p⁻¹, 1; n ≤ 12 for h = p | exact (Gröbner bases over Q for `thm:sing`) |
| `check_08_nakayama.py` | `prop:CY` (eq:nakayama) via the Koszul dual G^! | all admissible N ≤ 12 | over F_P |

A few notes on what "checked" means in specific places.

**Dimension of the center (`thm:gencenter`(i)).** For total degree d ≤ D the script computes dim Z(G)_d as the kernel of `a ↦ ([a,x1],[a,x2],[a,y1],[a,y2])` on G_d and compares it with the number of points of S of degree d. The lower bound comes from the lifts Λ_m, which are checked to be central exactly, with f an indeterminate. The upper bound is a rank computed after reducing Z[ζ,f] → F_P (P ≈ 10⁹, ζ ↦ a primitive N-th root of unity mod P, f ↦ 3). Reduction can only lower a rank, so this is a valid upper bound both for f an indeterminate and for f = 3. When the two bounds agree, the dimension is proved for those d.

**Gorenstein (`thm:gencenter`(iii)).** For k[S] three things are compared: palindromicity of the h-polynomial (Stanley's criterion), membership of (g0,g0,g0,g0) in L_gr, and the condition h ∈ ⟨-1,p⟩. The passage from k[S] to Z(G) is the Hilbert series argument of the paper, checked in low degrees as above.

**Nakayama (`prop:CY`).** This one works over F_P, not over Q(ζ). It uses the formula ν_G|_{G_1} = (−1)^{d+1}(ν_{G^!}|_{G^!_1})^* for Koszul AS-regular algebras. The sign convention is pinned down by the exterior algebra, where ν = −id. The result is consistent with eq:nakayama up to the inversion that comes from the order of the pairing.

**The checks have to be able to fail.** Many statements are "iff" (Q_m central iff h^m = 1, x1C central iff q² = 1, Gorenstein iff h ∈ ⟨-1,p⟩), so both outcomes occur in the runs. `check_01` also has explicit negative controls.

## What is not checked

These are proved in the paper by arguments that are not computations, and the scripts do not touch them:

- `prop:autgr` and `lem:normal1` (graded automorphisms, normal elements of degree one);
- the fact that every ozone automorphism is graded (the first step of `thm:ozall`, which uses [CGWZ25, Lemma 1.9]);
- `prop:maximal-order`, `prop:primitive`, `cor:PIiff`, `cor:noregular`, `prop:notpoly`, `prop:notriangular`;
- normality, Cohen–Macaulayness and the Krull dimension in `thm:gencenter`(ii), and the kernel statements in Section 7 beyond the Hilbert series comparison.

## Running

```
pip install -r requirements.txt
python run_all.py            # everything, about 3 minutes
python run_all.py ozone      # only scripts whose name contains "ozone"
python checks/check_05_ozone.py
```

Each script prints one line per statement with the number of cases passed, and exits with a nonzero status if any check fails. `run_all.py` writes the logs to `results/`. The logs of the run behind the paper are committed there, with the versions used in `results/SUMMARY.txt`.

## Layout

```
g14641/rings.py     Q(zeta_N)[f] (exact, f generic) and Q[p,h,f]
g14641/algebra.py   G(p,f,h) through the PBW normal form; Dickson polynomials
g14641/torus.py     the quantum torus on x1, T, y1, y2
g14641/lattice.py   Omega, the lattice L, PI degree, Hilbert bases, ozone groups
g14641/setup.py     b, c, alpha, Q_k, C, Lambda_m and the characters of lem:normal
checks/             one script per section of the paper
results/            logs
```

## Citation

If you use this code, please cite the paper and this repository (see `CITATION.cff`).
