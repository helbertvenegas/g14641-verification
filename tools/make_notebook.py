"""Builds explore.ipynb.  Run: python tools/make_notebook.py"""

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Explore the algebra $G(p,f,h)$ of type (14641)

Companion notebook for J. Gómez and H. Venegas, *PI degree, center and ozone group of the double extension family $G$ of type $(14641)$*.
Code: [github.com/helbertvenegas/g14641-verification](https://github.com/helbertvenegas/g14641-verification), archived at Zenodo, [doi:10.5281/zenodo.23935418](https://doi.org/10.5281/zenodo.23935418).

Pick the parameters in the second cell and use **Runtime → Run all**. The notebook gives the PI degree, the ozone group, generators of the center with their explicit form, and the Hilbert series of $Z(G)$.

**Everything is exact.** Coefficients live in $\mathbb Q(\zeta_N)[f]$ with $f$ an indeterminate, and the centrality of each generator is checked by multiplying in $G$ from the defining relations. The proofs in the paper do not depend on any of this.

Notation as in the paper: $\zeta$ is a primitive $N$-th root of unity, $p=\zeta^r$, $h=\zeta^s$, $q=hp$, $e=\operatorname{ord}(p^2)$, $Q_k=D_k(x_2,\alpha x_1^2)$ (Dickson polynomial), $C=x_2y_1y_2-x_1(cy_2^2+p^{-1}by_1^2)$."""))

cells.append(nbf.v4.new_code_cell(r'''#@title Setup (run once; about 30 seconds the first time)
import os, sys, subprocess
try:
    import flint, sympy, numpy  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-flint", "sympy", "numpy"], check=True)
if os.path.isdir("g14641"):
    REPO = "."
else:
    if not os.path.isdir("g14641-verification"):
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/helbertvenegas/g14641-verification.git"], check=True)
    REPO = "g14641-verification"
sys.path.insert(0, os.path.abspath(REPO))
from IPython.display import Markdown, display
from g14641.explore import *
print("ready")'''))

cells.append(nbf.v4.new_code_cell(r'''#@title Parameters
#@markdown **Option 1.** One of the four cases of the paper: $p$ a primitive $n$-th root of unity.
mode = "case of the paper"  #@param ["case of the paper", "general (N, r, s)"]
n = 6  #@param {type:"integer"}
h = "p^-1"  #@param ["p^-1", "1", "p", "-p^-1"]
#@markdown **Option 2.** General parameters: $\zeta$ a primitive $N$-th root of unity, $p=\zeta^r$, $h=\zeta^s$, with $\gcd(r,s,N)=1$ and $p^2\neq 1$.
N = 8  #@param {type:"integer"}
r = 2  #@param {type:"integer"}
s = 1  #@param {type:"integer"}

if mode == "case of the paper":
    N, r, s = from_case(n, h)
N, r, s = check_params(N, r, s)
display(Markdown(f"Working with $N={N}$, $p=\\zeta^{{{r}}}$, $h=\\zeta^{{{s}}}$."))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Invariants
PI degree (Theorem 3.6), Calabi–Yau and Nakayama (Proposition 2.5), Gorenstein criterion (Theorem 6.4), ozone group (Theorem 5.4)."""))

cells.append(nbf.v4.new_code_cell(r'''S = summary(N, r, s)
def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)
display(Markdown("| invariant | value |\n|---|---|\n" + "\n".join(f"| {k} | {fmt(v)} |" for k, v in S.items())))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Ozone group
$\varphi_{\xi,\theta,\eta}$ sends $(x_1,x_2,y_1,y_2)\mapsto(\xi x_1,\theta\xi x_2,\eta y_1,\theta\eta y_2)$. The group is computed from a basis of the lattice $\mathbb L$, through condition (5.2)."""))

cells.append(nbf.v4.new_code_cell(r'''G, struct, elems = ozone_latex(N, r, s)
display(Markdown(f"$\\operatorname{{Oz}}(G)\\cong {struct}$, of order {len(G)}."))
if elems:
    display(Markdown("Elements: " + ", ".join(f"${x}$" for x in elems)))
else:
    display(Markdown("(too many elements to list)"))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Generators of the center
For each $\mathbf m$ in the Hilbert basis of $S$, the lift $\Lambda_{\mathbf m}$ of (6.2) is central and its leading form is $x^{\mathbf m}$, so these elements generate $Z(G)$ (Theorem 6.4(i)).
The generator with leading monomial $x_2^{n}$ may show up as a power of $Q_e$ instead of $U=Q_n$; the two have the same leading form, so either one works (see the remark after Theorem 6.4).
With `check_centrality` on, each $\Lambda_{\mathbf m}$ is multiplied against $x_1,x_2,y_1,y_2$ in $G$. For $N$ above 12 or so this can take a while."""))

cells.append(nbf.v4.new_code_cell(r'''check_centrality = True  #@param {type:"boolean"}
st, rows = center_generators(N, r, s, check=check_centrality)
head = "| degree | generator | leading monomial |" + (" central | PBW terms |" if check_centrality else "")
sep = "|---|---|---|" + ("---|---|" if check_centrality else "")
lines = [head, sep]
for row in rows:
    line = f"| {row['degree']} | ${row['lift']}$ | ${row['leading']}$ |"
    if check_centrality:
        ok = row["central"] and row["leading_ok"]
        line += f" {'yes' if ok else '**NO**'} | {row['terms']} |"
    lines.append(line)
display(Markdown(f"$e={st.e}$. {len(rows)} generators:\n\n" + "\n".join(lines)))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Hilbert series of $Z(G)$
By total degree, written over the degrees of the generators above. For a hypersurface the numerator is $1-t^d$, with $d$ the degree of the relation."""))

cells.append(nbf.v4.new_code_cell(r'''coeffs, K, degs, pal = hilbert_series(N, r, s)
display(Markdown(f"${series_latex(K, degs)}$"))
display(Markdown(f"$\\dim Z(G)_d$ for $d=0,\\dots,{len(coeffs)-1}$: {coeffs}"))
display(Markdown(f"Numerator of the Hilbert series palindromic (by Stanley's theorem, the same as Gorenstein): {'yes' if pal else 'no'}"))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Your own computations
`A = st.alg` is the algebra $G$ for the chosen parameters, with $f$ an indeterminate. Elements are dictionaries of PBW monomials.
Useful operations: `A.mul(a, b)`, `A.add(a, b)`, `A.sub(a, b)`, `A.power(a, k)`, `A.comm(a, b)`, `A.is_central(a)`, `A.leading_form(a)`.
Ready-made elements: `st.x1, st.x2, st.y1, st.y2, st.C, st.Q(k), st.U(), st.Y(1), st.Y(2), st.Lambda(m)`.
In printed coefficients, `ζ` is $\zeta$ and `f` is $f$."""))

cells.append(nbf.v4.new_code_cell(r'''A = st.alg
x1, x2, y1, y2, C = st.x1, st.x2, st.y1, st.y2, st.C
Qe = st.Q(st.e)

print("C =", element_str(C))
print()
print("Q_e =", element_str(Qe))
print()
print("Is C central?", A.is_central(C))
print("Is x1*C central?", A.is_central(A.mul(x1, C)), " (Theorem 5.8: iff G is Calabi-Yau)")
print("y1^e Q_e - h^e Q_e y1^e = 0?", A.is_zero(A.sub(A.mul(A.power(y1, st.e), Qe), A.scale(st.R.zeta(st.s * st.e * st.e), A.mul(Qe, A.power(y1, st.e))))))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## Full verification (optional, about 3 minutes)
Runs every check of the repository (about 150,000 checks). Each line should say PASS."""))

cells.append(nbf.v4.new_code_cell(r'''run_all = False  #@param {type:"boolean"}
if run_all:
    subprocess.run([sys.executable, os.path.join(REPO, "run_all.py")])'''))

nb["cells"] = cells
nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
out = os.path.join(HERE, "..", "explore.ipynb")
nbf.write(nb, out)
print("wrote", os.path.abspath(out))
