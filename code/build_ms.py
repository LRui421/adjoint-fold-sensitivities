"""build_ms -- assemble main.tex from the three parts, fill @@tokens@@ from tokens.json, build the two data tables
from tables2/per_network.csv, and compile with pdflatex + bibtex (portable MiKTeX)."""
import os, re, sys, json, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, "ms")
GORDER = ["Mut-ER", "Mut-SF", "Mut-SF-het", "Webs-proj", "Webs-bip", "Reg-ER", "Reg-SF", "Reg-SF-het", "Reg-dir", "RegB-ER", "RegB-SF"]


def num(x, nd):
    if not np.isfinite(x):
        return "--"
    t = f"{x:.{nd}f}"
    return f"${t}$" if t.startswith("-") else t


def tables(T1):
    rows_c, rows_r = [], []
    for g in GORDER:
        s = T1[T1.group == g]
        if not len(s):
            continue
        nr = int(s.rec_ok.fillna(0).sum())
        rows_c.append(" & ".join([g, str(len(s)), num(s.N.mean(), 1), num(s.beta_c.mean(), 3), num(s.a.median(), 4),
                                  num(s.gap.min(), 2), num(s.kappa.median(), 2), num(s.PR_u.mean(), 2), str(nr),
                                  num(s.hyst.mean(), 2) if nr else "--"]) + r"\\")
        rows_r.append(" & ".join([g, num(100 * s.err_gbb.mean(), 1), num(100 * s.err_spec.mean(), 1), num(100 * s.err_closure.mean(), 1),
                                  num(s.rho_GBB_gradient.mean(), 2), num(s.rho_closure_exact.mean(), 2), num(s.rho_RC1.mean(), 2),
                                  num(s.rho_RC2.mean(), 2), num(100 * s.frac_neg_gbb.mean(), 1)]) + r"\\")
    return "\n".join(rows_c), "\n".join(rows_r)


def fill(txt, tok):
    missing = sorted(set(re.findall(r"@@([\w.]+)@@", txt)) - set(tok))
    if missing:
        raise KeyError(f"missing tokens: {missing}")
    return re.sub(r"@@([\w.]+)@@", lambda m: tok[m.group(1)], txt)


def assemble(tok):
    T1 = pd.read_csv(os.path.join(HERE, "tables2", "per_network.csv"))
    tc, tr = tables(T1)
    tok = dict(tok)
    tok["TABLE_CERT"], tok["TABLE_RED"] = tc, tr
    abs_ = json.load(open(os.path.join(MS, "abstract.json")))["ABSTRACT"]
    tok["ABSTRACT"] = re.sub(r"\$([^$]+)\$", lambda m: "{\\unboldmath$" + m.group(1) + "$}", fill(abs_, tok))
    parts = [open(os.path.join(MS, f), encoding="utf8").read() for f in ("part1_front_methods.tex", "part2_results.tex", "part3_appendix.tex")]
    txt = "\n".join(parts) + "\n\\bibliography{refs}\n\n\\end{document}\n"
    txt = fill(txt, tok)
    open(os.path.join(MS, "main.tex"), "w", encoding="utf8").write(txt)
    return txt


def compile_pdf(env, bindir, main="main"):
    logs = []
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        exe = os.path.join(bindir, cmd[0] + ".exe")
        args = [exe] + (["--disable-installer", "-interaction=nonstopmode", main + ".tex"] if cmd[0] == "pdflatex" else [main])
        p = subprocess.run(args, cwd=MS, env=env, capture_output=True, text=True, errors="ignore", timeout=600)
        logs.append((cmd[0], p.returncode))
    log = open(os.path.join(MS, main + ".log"), errors="ignore").read()
    issues = [l for l in log.splitlines() if l.startswith("!") or "Warning: Citation" in l or "Warning: Reference" in l
              or "undefined" in l.lower() or "Missing character" in l]
    over = [l for l in log.splitlines() if l.startswith("Overfull")]
    return logs, issues, over
