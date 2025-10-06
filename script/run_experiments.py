#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, tempfile, re, csv, os, sys, glob, shutil, math
from datetime import datetime
from collections import defaultdict

# ==============================================================
# Inyección de estrategia de búsqueda (model-aware)
# ==============================================================

# Regex para localizar la primera línea 'solve ... satisfy;'
SOLVE_REGEX = re.compile(
    r"(?m)^[ \t]*(?!%)[ \t]*solve\s*(::[^\n]*)?\s*satisfy\s*;",
    re.IGNORECASE
)

# (Opcional) Etiqueta explícita de modelo en la cabecera:
#   % MODEL_ID: sudoku
#   % MODEL_ID: reunion
MODEL_ID_RE = re.compile(r"^\s*%+\s*MODEL_ID:\s*(\w+)", re.IGNORECASE | re.MULTILINE)

# Heurísticas para detectar tipo de modelo cuando no hay etiqueta
HINTS = {
    "sudoku": {
        "must": [re.compile(r"\barray\s*\[\s*S\s*,\s*S\s*\]\s*of\s*var\b.*:\s*X\b", re.DOTALL)],
        "any":  [re.compile(r"\bG\s*;\s*$", re.MULTILINE),
                 re.compile(r"\ball_different\(\s*\[\s*X\[r,c\]")],
    },
    "reunion": {
        "must": [re.compile(r"\barray\s*\[\s*S\s*\]\s*of\s*var\s*POS\s*:\s*POS_OF\b"),
                 re.compile(r"\barray\s*\[\s*POS\s*\]\s*of\s*var\s*S\s*:\s*PER_AT\b"),
                 re.compile(r"\binverse\s*\(\s*POS_OF\s*,\s*PER_AT\s*\)")],
        "any":  [re.compile(r"\bNEXT\b"), re.compile(r"\bSEP\b"), re.compile(r"\bDIST\b")],
    },
}

# ¿El modelo ya define una lista de branching?
HAS_DECISION = re.compile(r"\bDECISION_VARS\b")
HAS_BRANCH   = re.compile(r"\bBRANCH_VARS\b")

def detect_model_kind(model_text: str) -> str:
    """Detecta el tipo de modelo (sudoku|reunion|unknown) por etiqueta o heurística."""
    m = MODEL_ID_RE.search(model_text)
    if m:
        tag = m.group(1).strip().lower()
        if tag in ("sudoku", "reunion"):
            return tag
    for kind, patt in HINTS.items():
        if all(rx.search(model_text) for rx in patt["must"]) and any(rx.search(model_text) for rx in patt["any"]):
            return kind
    return "unknown"

def pick_existing_branch_name(model_text: str) -> str | None:
    """Devuelve el nombre de la lista de branching si ya existe en el modelo."""
    if HAS_DECISION.search(model_text): return "DECISION_VARS"
    if HAS_BRANCH.search(model_text):   return "BRANCH_VARS"
    return None

def ensure_branch_vars(model_text: str, kind: str) -> tuple[str, str | None]:
    """
    Asegura una lista 1D con variables de ramificación.
    - Reusa DECISION_VARS/BRANCH_VARS si existen.
    - Para 'sudoku' evita depender de S/DIG: usa index_set_1_of_2/2_of_2 y 'var int'.
    - Inserta justo antes del 'solve'.
    """
    name = pick_existing_branch_name(model_text)
    if name:
        return model_text, name

    if kind == "sudoku":
        # Robustez: no asumimos S ni DIG; opcionalmente filtramos por G[r,c]==0 si G existe.
        snippet = (
            "% === Injected by runner: DECISION_VARS (Sudoku) ===\n"
            "array[int] of var int: DECISION_VARS =\n"
            "  [ X[r,c] |\n"
            "    r in index_set_1_of_2(X),\n"
            "    c in index_set_2_of_2(X)\n"
            "    where (if exists(index_set_1_of_2(G)) then G[r,c] = 0 else true endif)\n"
            "  ];\n"
        )
        name = "DECISION_VARS"

    elif kind == "reunion":
        snippet = (
            "% === Injected by runner: DECISION_VARS (Reunión) ===\n"
            "array[int] of var int: DECISION_VARS = [ POS_OF[p] | p in S ];\n"
        )
        name = "DECISION_VARS"

    else:
        # Desconocido: no sabemos extraer decisiones de forma segura
        return model_text, None

    m = SOLVE_REGEX.search(model_text)
    if not m:
        return model_text, None

    i = m.start()
    patched = model_text[:i] + snippet + model_text[i:]
    return patched, name

# ---- Plantillas de 'solve' separadas por modelo ----
# Clave especial "default": None => no reemplaza el 'solve' del modelo
SOLVE_TEMPLATES_BY_MODEL = {
    "sudoku": {
        "ff_min":     "solve :: int_search({VARS}, first_fail,  indomain_min,   complete) satisfy;",
        "wdeg_split": "solve :: int_search({VARS}, dom_w_deg,   indomain_split, complete) satisfy;",
        "default":    None,
    },
    "reunion": {
        "ff_min":     "solve :: int_search({VARS}, first_fail,  indomain_min,   complete) satisfy;",
        "wdeg_split": "solve :: int_search({VARS}, dom_w_deg,   indomain_split, complete) satisfy;",
        "default":    None,
    },
    "unknown": {
        "ff_min":     "solve :: int_search({VARS}, first_fail,  indomain_min,   complete) satisfy;",
        "wdeg_split": "solve :: int_search({VARS}, dom_w_deg,   indomain_split, complete) satisfy;",
        "default":    None,
    },
}

# Aliases aceptados para compatibilidad (p. ej., si venías usando 'domdeg_split')
STRAT_ALIASES = {
    "domdeg_split": "wdeg_split",
}

def normalize_strategy_name(s: str) -> str:
    s = s.strip().lower()
    return STRAT_ALIASES.get(s, s)

def inject_solve_by_kind(model_text: str, strategy: str, kind: str) -> str:
    """
    Prepara el modelo para la estrategia pedida, detectando tipo de problema y
    garantizando la lista de branching apropiada. Si no puede, cae a 'solve satisfy;'.
    """
    strategy = normalize_strategy_name(strategy)
    templates = SOLVE_TEMPLATES_BY_MODEL.get(kind, SOLVE_TEMPLATES_BY_MODEL["unknown"])
    if strategy not in templates:
        # estrategia desconocida -> no tocar el solve
        return model_text

    tmpl = templates[strategy]
    if tmpl is None:
        # 'default': no tocar el solve del modelo
        return model_text

    # Asegurar la lista de branching apropiada para el modelo
    txt, varname = ensure_branch_vars(model_text, kind)
    if varname is None:
        # No sabemos con qué ramificar -> reemplazamos con solve satisfy;
        return SOLVE_REGEX.sub("solve satisfy;", model_text, count=1)

    if not SOLVE_REGEX.search(txt):
        raise RuntimeError("Could not find a 'solve ... satisfy;' line to replace.")

    solve_line = tmpl.format(VARS=varname)
    return SOLVE_REGEX.sub(solve_line, txt, count=1)

# ==============================================================
# Utilidades varias (estadísticas, selección, tablas)
# ==============================================================

def format_time_sci(t, digits=3):
    if t is None:
        return None
    return f"{t:.{digits}e}"

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
                stats["status"] = kv.strip()

    # Casts
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
    Normaliza el tiempo total:
    - Si 'time' existe, usarlo.
    - Si no, usar initTime + solveTime si existen.
    - Si nada existe, None.
    """
    t = stats.get("time")
    if isinstance(t, (int, float)):
        return t
    it = stats.get("initTime")
    st = stats.get("solveTime")
    if isinstance(it, (int, float)) or isinstance(st, (int, float)):
        return (it or 0.0) + (st or 0.0)
    return None

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
        # 1) extremos por instancia (best/worst time)
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

        # 2) Frontera de Pareto en (time_raw, nodes, failures)
        idxs = pareto_min(group, keys=["time_raw", "nodes", "failures"])
        for i in idxs:
            shortlisted.append((group[i], "pareto-front"))

        # 3) Outliers (IQR) en time_raw y nodes
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
        # formateo de tiempo si viene en float
        try:
            if isinstance(vals[4], float):
                vals[4] = f"{vals[4]:.3f}"
        except Exception:
            pass
        line = " & ".join([str(v) if v is not None else "" for v in vals]) + " \\\\"
        lines.append("    " + line)
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ==============================================================
# Main
# ==============================================================

def main():
    ap = argparse.ArgumentParser()
    # ÚNICO base para rutas de salida
    ap.add_argument("--base-dir", required=True, help="Directorio base donde se guardará todo lo generado")
    # Barrido
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--solver", nargs="+", default=["gecode", "chuffed"], help="Lista de solvers (e.g., gecode chuffed)")
    # Tres estrategias por defecto (idénticas para ambos solvers)
    ap.add_argument("--strategy", nargs="+", default=["ff_min", "wdeg_split", "default"],
                    help="Estrategias: ff_min | wdeg_split (alias: domdeg_split) | default")
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
    model_kind = detect_model_kind(model_text)

    # Validar y normalizar estrategias
    norm_strategies = []
    for s in args.strategy:
        ns = normalize_strategy_name(s)
        # permitir solo las 3 soportadas
        if ns not in ("ff_min", "wdeg_split", "default"):
            print(f"[WARN] estrategia '{s}' no soportada; se ignora.", file=sys.stderr)
            continue
        norm_strategies.append(ns)
    if not norm_strategies:
        norm_strategies = ["ff_min", "wdeg_split", "default"]

    rows = []
    runs_dir = rel("results/artifacts")
    os.makedirs(runs_dir, exist_ok=True)
    fzn_dir = rel("results/flat")
    os.makedirs(fzn_dir, exist_ok=True)

    for solver in args.solver:
        for strat in norm_strategies:
            # Genera una versión temporal del modelo con la estrategia (si aplica)
            try:
                mod_txt = inject_solve_by_kind(model_text, strat, model_kind)
            except Exception as e:
                print(f"[WARN] Strategy {strat}: {e}; using default solve.", file=sys.stderr)
                mod_txt = model_text

            # --- DEBUG: muestra la línea 'solve' resultante (previa a compilar) ---
            try:
                msolve = SOLVE_REGEX.search(mod_txt)
                solve_preview = mod_txt[msolve.start():msolve.end()] if msolve else "<no-solve>"
                print(f"[DEBUG] {solver}/{strat}: solve-line => {solve_preview}")
            except Exception:
                pass

            with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tmpm:
                tmpm.write(mod_txt)
                tmpm.flush()
                tmpm_path = tmpm.name

            try:
                for data_path in data_files:
                    base = os.path.splitext(os.path.basename(data_path))[0]
                    tag  = f"{base}__{solver}__{strat}"

                    # (Preview opcional) contar DECISION_VARS si existe
                    if "DECISION_VARS" in mod_txt:
                        try:
                            with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tmpc:
                                tmpc.write(mod_txt + "\n")
                                tmpc.write('output ["#DECISION_VARS=", show(length(DECISION_VARS)), "\\n"];')
                                tmpc.flush()
                                preview_proc = subprocess.run(
                                    ["minizinc", "--solver", solver, "--time-limit", "1000", tmpc.name, data_path],
                                    capture_output=True, text=True
                                )
                                if preview_proc.returncode == 0:
                                    mcount = re.search(r"#DECISION_VARS=\s*(\d+)", preview_proc.stdout)
                                    if mcount:
                                        print(f"[DEBUG] {solver}/{strat}/{base}: DECISION_VARS size = {mcount.group(1)}")
                        except Exception:
                            pass

                    # ----------------------------------------------------------------
                    # Compilar a .fzn para inspeccionar 'branch' y luego ejecutar
                    # ----------------------------------------------------------------
                    fzn_path = os.path.join(fzn_dir, f"{tag}.fzn")
                    cmd_compile = ["minizinc", "--solver", solver, "-c", tmpm_path, data_path, "-o", fzn_path]
                    proc_c = subprocess.run(cmd_compile, capture_output=True, text=True)
                    stdout_c, stderr_c, rc_c = proc_c.stdout, proc_c.stderr, proc_c.returncode

                    # Si la compilación falla por identificador indefinido, reintenta con solve por defecto
                    if rc_c != 0 and ("undefined identifier" in (stdout_c + stderr_c).lower()):
                        print(f"[WARN] {solver}/{strat}: retrying (compile) with default solve.", file=sys.stderr)
                        mod_txt2 = inject_solve_by_kind(model_text, "default", model_kind)
                        with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tmpm2:
                            tmpm2.write(mod_txt2)
                            tmpm2.flush()
                            tmpm_path2 = tmpm2.name
                        fzn_path = os.path.join(fzn_dir, f"{tag}.fallback.fzn")
                        cmd_compile = ["minizinc", "--solver", solver, "-c", tmpm_path2, data_path, "-o", fzn_path]
                        proc_c = subprocess.run(cmd_compile, capture_output=True, text=True)
                        stdout_c, stderr_c, rc_c = proc_c.stdout, proc_c.stderr, proc_c.returncode

                    # Loguear branch-lines del FlatZinc
                    if os.path.exists(fzn_path):
                        try:
                            with open(fzn_path, "r", encoding="utf-8", errors="ignore") as ff:
                                branch_lines = [ln.strip() for ln in ff if ln.startswith("branch")]
                            if branch_lines:
                                print(f"[DEBUG] branch-lines ({solver}/{strat}/{base}):")
                                for ln in branch_lines[:8]:
                                    print("   ", ln)
                        except Exception:
                            pass

                    # Ejecutar el fzn con estadísticas
                    cmd = ["minizinc", "--solver", solver, "--statistics", "--time-limit", str(args.time_limit), fzn_path]
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode

                    # Fallback en ejecución si hubo error
                    if rc != 0 and ("undefined identifier" in (stdout + stderr).lower()):
                        print(f"[WARN] {solver}/{strat}: retrying (run) with default solve.", file=sys.stderr)
                        mod_txt2 = inject_solve_by_kind(model_text, "default", model_kind)
                        with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False) as tmpm2:
                            tmpm2.write(mod_txt2)
                            tmpm2.flush()
                            tmpm_path2 = tmpm2.name
                        fzn_path2 = os.path.join(fzn_dir, f"{tag}.runfallback.fzn")
                        cmd_compile2 = ["minizinc", "--solver", solver, "-c", tmpm_path2, data_path, "-o", fzn_path2]
                        subprocess.run(cmd_compile2, capture_output=True, text=True)
                        cmd = ["minizinc", "--solver", solver, "--statistics", "--time-limit", str(args.time_limit), fzn_path2]
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode

                    # Guardar salidas
                    with open(os.path.join(runs_dir, f"{tag}.sol.txt"), "w", encoding="utf-8") as fsol:
                        fsol.write(stdout)
                    with open(os.path.join(runs_dir, f"{tag}.log.txt"), "w", encoding="utf-8") as flog:
                        flog.write("CMD: " + " ".join(cmd) + "\n\n")
                        flog.write(stdout + "\n\n--- STDERR ---\n" + stderr)

                    # Parseo stats
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

        # Copiar artefactos de la shortlist
        if args.copy_shortlist_to:
            target_dir = rel(args.copy_shortlist_to)
            os.makedirs(target_dir, exist_ok=True)
            copied = 0
            for r in sl:
                tag = f"{r['file']}__{r['solver']}__{r['strategy']}"
                for ext in [".log.txt", ".sol.txt"]:
                    src = os.path.join(rel("results"), "artifacts", f"{tag}{ext}")
                    if os.path.exists(src):
                        dst = os.path.join(target_dir, f"{tag}{ext}")
                        shutil.copy2(src, dst)
                        copied += 1
            print(f"Copied {copied} artifacts to {target_dir}")

if __name__ == "__main__":
    main()
