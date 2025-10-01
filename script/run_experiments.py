#!/usr/bin/env python3
import argparse, subprocess, tempfile, re, csv, os, sys, glob, shutil, math
from datetime import datetime
from collections import defaultdict

# ---------- Plantillas de solve ----------
SOLVE_TEMPLATES = {
    "ff_min":        "solve :: int_search(BRANCH_VARS, first_fail, indomain_min, complete) satisfy;",
    "domdeg_split":  "solve :: int_search(BRANCH_VARS, dom_w_deg,  indomain_split, complete) satisfy;",
    "input_min":     "solve :: int_search(BRANCH_VARS, input_order, indomain_min, complete) satisfy;",
    "default":       None,
}
SOLVE_REGEX = re.compile(r"solve\s*(::\s*.*)?\s*satisfy\s*;", re.IGNORECASE | re.DOTALL)

def inject_solve(model_text: str, strategy: str) -> str:
    tmpl = SOLVE_TEMPLATES[strategy]
    if tmpl is None:
        return model_text
    if not SOLVE_REGEX.search(model_text):
        raise RuntimeError("Could not find a 'solve ... satisfy;' line to replace.")
    return SOLVE_REGEX.sub(tmpl, model_text, count=1)

def format_time_sci(t, digits=3):
    if t is None:
        return None
    return f"{t:.{digits}e}"

# ---------- Parseo de estadísticas ----------
def parse_stats(mzn_text: str):
    """
    Lee líneas '%%%mzn-stat:' y '%%%mzn-status' desde stdout+stderr combinados.
    Devuelve dict con casts básicos.
    """
    stats = {}
    for line in mzn_text.splitlines():
        if line.startswith("%%%mzn-stat:") or line.startswith("%%%mzn-status"):
            kv = line.split(":", 1)[1].strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k.strip()] = v.strip()
            else:
                # Algunas builds pueden emitir sólo el status sin '='
                stats["status"] = kv.strip()

    # Cast
    def cast_float(k):
        if k in stats:
            try:
                stats[k] = float(stats[k])
            except:
                pass

    def cast_int(k):
        if k in stats:
            try:
                stats[k] = int(float(stats[k]))
            except:
                pass

    for k in ["time", "initTime", "solveTime"]:
        cast_float(k)
    for k in ["nodes", "failures", "solutions", "restarts", "peakDepth", "variables", "constraints"]:
        cast_int(k)

    return stats

def compute_total_time(stats):
    """
    Normaliza el tiempo:
    - Si 'time' existe, úsalo.
    - Si no, suma initTime + solveTime si al menos uno existe.
    - Si nada existe, devuelve None.
    """
    t = stats.get("time")
    if isinstance(t, (int, float)):
        return t
    it = stats.get("initTime")
    st = stats.get("solveTime")
    if isinstance(it, (int, float)) or isinstance(st, (int, float)):
        return (it or 0.0) + (st or 0.0)
    return None

# ---------- Selección de interesantes ----------
def pareto_min(rows, keys):
    idxs = []
    for i, ri in enumerate(rows):
        if any(ri.get(k) is None for k in keys):
            continue
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            if any(rj.get(k) is None for k in keys):
                continue
            better_or_equal_all = all(rj[k] <= ri[k] for k in keys)
            strictly_better_one = any(rj[k] < ri[k] for k in keys)
            if better_or_equal_all and strictly_better_one:
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs

def iqr_fences(values, k=1.5):
    if len(values) < 4:
        return (None, None)
    vs = sorted(values)
    mid = len(vs) // 2
    lower = vs[:mid]
    upper = vs[mid + 1:] if len(vs) % 2 == 1 else vs[mid:]
    def median(a):
        m = len(a) // 2
        return (a[m] + a[~m]) / 2 if len(a) % 2 == 0 else a[m]
    q1 = median(lower) if lower else None
    q3 = median(upper) if upper else None
    if q1 is None or q3 is None:
        return (None, None)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def shortlist_from_rows(rows, topk=2, delta=1.5, iqr_k=1.5):
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    shortlisted = []

    for f, group in by_file.items():
        # 1) extremos por instancia (best/worst time) usando time_raw (float)
        g_time = [r for r in group if r.get("time_raw") is not None]
        if g_time:
            best = sorted(
                g_time,
                key=lambda r: (r.get("time_raw", float("inf")),
                               r.get("nodes") if r.get("nodes") is not None else math.inf)
            )[:topk]
            worst = sorted(
                g_time,
                key=lambda r: (-(r.get("time_raw", -float("inf"))),
                               -(r.get("nodes") if r.get("nodes") is not None else -1))
            )[:topk]
            for r in best:
                shortlisted.append((r, "best-time"))
            for r in worst:
                shortlisted.append((r, "worst-time"))

        # 2) Pareto en (time_raw, nodes, failures)
        idxs = pareto_min(group, keys=["time_raw", "nodes", "failures"])
        for i in idxs:
            shortlisted.append((group[i], "pareto-front"))

        # 3) Outliers IQR en time_raw y nodes
        for metric in ["time_raw", "nodes"]:
            vals = [r[metric] for r in group if r.get(metric) is not None]
            lo, hi = iqr_fences(vals, k=iqr_k)
            if lo is None:
                continue
            for r in group:
                v = r.get(metric)
                if v is None:
                    continue
                if v < lo or v > hi:
                    shortlisted.append((r, f"outlier-{metric}"))

        # 5) Anomalías: rc != 0 o solutions != 1
        for r in group:
            if (r.get("rc") not in (0, None)) or (r.get("solutions") not in (1, None)):
                shortlisted.append((r, "anomaly"))

        # 4) Desacuerdo entre solvers (misma estrategia)
        by_strat = defaultdict(list)
        for r in group:
            by_strat[r["strategy"]].append(r)
        for strat, g in by_strat.items():
            times = [r["time_raw"] for r in g if r.get("time_raw") is not None]
            nodes = [r["nodes"] for r in g if r.get("nodes") is not None]

            def ratio(vs):
                return max(vs) / min(vs) if len(vs) >= 2 and min(vs) > 0 else 1.0

            if times and ratio(times) >= delta:
                tmin = min(g, key=lambda r: r.get("time_raw", float("inf")))
                tmax = max(g, key=lambda r: r.get("time_raw", -1.0))
                shortlisted.append((tmin, f"solver-gap-time({ratio(times):.2f}x)"))
                shortlisted.append((tmax, f"solver-gap-time({ratio(times):.2f}x)"))

            if nodes and ratio(nodes) >= delta:
                nmin = min(g, key=lambda r: r.get("nodes", float("inf")))
                nmax = max(g, key=lambda r: r.get("nodes", -1))
                shortlisted.append((nmin, f"solver-gap-nodes({ratio(nodes):.2f}x)"))

    # De-duplicar
    seen = set()
    out_rows = []
    for r, reason in shortlisted:
        key = (r["file"], r["solver"], r["strategy"])
        if key in seen:
            continue
        seen.add(key)
        o = dict(r)
        o["reason"] = reason
        out_rows.append(o)
    return out_rows

def emit_latex_table(rows, path, caption="Shortlist de corridas interesantes", label="tab:shortlist"):
    cols = ["file", "solver", "strategy", "reason", "time", "nodes", "failures", "peakDepth", "solutions", "status"]
    header = ["Archivo", "Solver", "Estrategia", "Motivo", "Tiempo (s)", "Nodes", "Failures", "Depth", "Sol.", "Status"]
    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{l l l l r r r r r l}")
    lines.append("    \\hline")
    lines.append("    " + " & ".join([f"\\textbf{{{h}}}" for h in header]) + " \\\\")
    lines.append("    \\hline")
    for r in rows:
        vals = [r.get(k, "") for k in cols]
        if isinstance(vals[4], float):
            vals[4] = f"{vals[4]:.3f}"
        line = " & ".join([str(v) if v is not None else "" for v in vals]) + " \\\\"
        lines.append("    " + line)
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    # ÚNICO base para rutas de salida
    ap.add_argument("--base-dir", required=True, help="Directorio base donde se guardará todo lo generado")
    # Barrido
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--solver", nargs="+", default=["chuffed"])
    ap.add_argument("--strategy", nargs="+", default=["ff_min", "domdeg_split", "input_min", "default"])
    ap.add_argument("--time-limit", type=int, default=60000)
    # Rutas (relativas a base-dir por defecto)
    ap.add_argument("--out", default="results/results.csv")
    # Selección
    ap.add_argument("--shortlist", action="store_true", default=True, help="(on) generar shortlist")
    ap.add_argument("--shortlist-out", default="results/shortlist.csv")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--iqr-k", type=float, default=1.5)
    ap.add_argument("--copy-shortlist-to", default="results/shortlist_artifacts", help="carpeta (relativa a base-dir) para copiar .log/.sol")
    args = ap.parse_args()

    # Normaliza rutas relativas a base-dir
    def rel(p):
        return p if os.path.isabs(p) else os.path.join(args.base_dir, p)

    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.dzn")))
    if not data_files:
        print("No .dzn files found in", args.data_dir, file=sys.stderr)
        sys.exit(2)

    os.makedirs(rel(os.path.dirname(args.out) or "."), exist_ok=True)

    with open(args.model, "r", encoding="utf-8") as f:
        model_text = f.read()

    rows = []
    for solver in args.solver:
        for strat in args.strategy:
            try:
                mod_txt = inject_solve(model_text, strat)
            except Exception as e:
                print(f"[WARN] Strategy {strat}: {e}; using default solve.", file=sys.stderr)
                mod_txt = model_text
            with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tmpm:
                tmpm.write(mod_txt)
                tmpm.flush()
                tmpm_path = tmpm.name

            try:
                for data_path in data_files:
                    base = os.path.splitext(os.path.basename(data_path))[0]
                    tag  = f"{base}__{solver}__{strat}"
                    cmd = [
                        "minizinc",
                        "--solver", solver,
                        "--statistics",
                        "--time-limit", str(args.time_limit),
                        tmpm_path,
                        data_path
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode

                    # escribir salidas en BASE/results
                    runs_dir = rel("results/artifacts")
                    os.makedirs(runs_dir, exist_ok=True)
                    with open(os.path.join(runs_dir, f"{tag}.sol.txt"), "w", encoding="utf-8") as fsol:
                        fsol.write(stdout)
                    with open(os.path.join(runs_dir, f"{tag}.log.txt"), "w", encoding="utf-8") as flog:
                        flog.write("CMD: " + " ".join(cmd) + "\n\n")
                        flog.write(stdout + "\n\n--- STDERR ---\n" + stderr)

                    # Parsear stdout+stderr juntos
                    stats = parse_stats(stdout + "\n" + stderr)

                    # Tiempo total normalizado
                    tsec_raw = compute_total_time(stats)
                    tsec = format_time_sci(tsec_raw, digits=3)

                    rows.append({
                        "file": base,
                        "solver": solver,
                        "strategy": strat,
                        "rc": rc,
                        "status": stats.get("status"),
                        "time_raw": tsec_raw,
                        "time": tsec,
                        "nodes": stats.get("nodes"),
                        "failures": stats.get("failures"),
                        "peakDepth": stats.get("peakDepth"),
                        "solutions": stats.get("solutions"),
                        "restarts": stats.get("restarts"),
                        "initTime": stats.get("initTime"),
                        "solveTime": stats.get("solveTime"),
                    })
                    print(f"[{solver}/{strat}] {base}: rc={rc}, status={stats.get('status')}, "
                          f"time={tsec}, nodes={stats.get('nodes')}, failures={stats.get('failures')}")
            finally:
                pass

    # Guardar CSV de resultados (en BASE)
    out_csv = rel(args.out)
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=["file","solver","strategy","rc","status","time_raw","time",
            "nodes","failures","peakDepth","solutions","restarts","initTime","solveTime"]

        )
        w.writeheader()
        w.writerows(rows)
    print("Saved results CSV:", out_csv)

    # Shortlist
    if args.shortlist:
        sl = shortlist_from_rows(rows, topk=args.topk, delta=args.delta, iqr_k=args.iqr_k)

        sl_csv = rel(args.shortlist_out)
        os.makedirs(os.path.dirname(sl_csv) or ".", exist_ok=True)
        with open(sl_csv, "w", newline="", encoding="utf-8") as fsl:
            w = csv.DictWriter(
                fsl,
                fieldnames=["file","solver","strategy","reason","time","nodes","failures",
            "peakDepth","solutions","restarts","rc","status","initTime","solveTime","time_raw"]
            )
            w.writeheader()
            for r in sl:
                w.writerow(r)
        print(f"Saved shortlist CSV: {sl_csv} ({len(sl)} rows)")

        # Copiar artefactos
        if args.copy_shortlist_to:
            target_dir = rel(args.copy_shortlist_to)
            os.makedirs(target_dir, exist_ok=True)
            copied = 0
            for r in sl:
                tag = f"{r['file']}__{r['solver']}__{r['strategy']}"
                for ext in [".log.txt", ".sol.txt"]:
                    src = os.path.join(rel("results"), tag + ext)
                    if os.path.exists(src):
                        dst = os.path.join(target_dir, tag + ext)
                        shutil.copy2(src, dst)
                        copied += 1
            print(f"Copied {copied} artifacts to {target_dir}")

if __name__ == "__main__":
    main()
