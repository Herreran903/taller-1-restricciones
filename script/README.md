# Barrido de pruebas para un modelo MiniZinc (genérico)
# Requisitos: MiniZinc en PATH y Python 3.x
# Modelo:     <ruta/al/modelo.mzn>              (p. ej., modelos/<PROBLEMA>/<modelo>.mzn)
# Instancias: <ruta/a/tests>/*.dzn              (p. ej., modelos/<PROBLEMA>/tests)

# (macOS) Asegurar MiniZinc en PATH
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"

# Barrido genérico (multi-solver / multi-estrategia) — macOS / Linux
python3 script/run_experiments.py \
  --base-dir modelos/<PROBLEMA> \
  --model    modelos/<PROBLEMA>/<modelo>.mzn \
  --data-dir modelos/<PROBLEMA>/tests \
  --solver   chuffed gecode \
  --strategy ff_min domdeg_split input_min default \
  --time-limit 60000


# =============================
# Versión Windows (PowerShell)
# =============================
# (Windows) Asegurar MiniZinc en PATH para la sesión actual
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
# (si tu instalación es "MiniZinc IDE" o está en otra carpeta, ajusta la ruta)

# Barrido genérico (multi-solver / multi-estrategia) — Windows PowerShell
python script\run_experiments.py `
  --base-dir modelos\<PROBLEMA> `
  --model    modelos\<PROBLEMA>\<modelo>.mzn `
  --data-dir modelos\<PROBLEMA>\tests `
  --solver   chuffed gecode `
  --strategy ff_min domdeg_split input_min default `
  --time-limit 60000

# (Alternativa en CMD con continuaciones)
rem set "PATH=C:\Program Files\MiniZinc;%PATH%"
python script\run_experiments.py ^
  --base-dir modelos\<PROBLEMA> ^
  --model    modelos\<PROBLEMA>\<modelo>.mzn ^
  --data-dir modelos\<PROBLEMA>\tests ^
  --solver   chuffed gecode ^
  --strategy ff_min domdeg_split input_min default ^
  --time-limit 60000


# -----------------------------
# Sudoku
# -----------------------------

# (macOS) MiniZinc en PATH
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"

# Barrido para Sudoku (un solver y una estrategia) — macOS / Linux
python3 script/run_experiments.py \
  --base-dir modelos/sodoku \
  --model    modelos/sodoku/sudoku.mzn \
  --data-dir modelos/sodoku/tests \
  --solver   gecode \
  --strategy ff_min \
  --time-limit 60000

# Barrido para Sudoku — Windows PowerShell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
python script\run_experiments.py `
  --base-dir modelos\sodoku `
  --model    modelos\sodoku\sudoku.mzn `
  --data-dir modelos\sodoku\tests `
  --solver   gecode `
  --strategy ff_min `
  --time-limit 60000


# -----------------------------
# Salidas (todas relativas a --base-dir)
# -----------------------------
# runs/results.csv              -> todas las corridas (incluye time_raw y time en notación científica)
# runs/shortlist.csv            -> corridas “interesantes” (mejores/peores por instancia, Pareto, outliers, gaps, anomalías)
# runs/shortlist_artifacts/     -> copia de *.log.txt y *.sol.txt de la shortlist
# runs/shortlist_table.tex      -> tabla LaTeX para el informe


# -----------------------------
# Tips rápidos
# -----------------------------
# - rc=0 solo indica que el proceso corrió bien; mira 'status' para SAT/UNSAT/UNKNOWN.
# - Si 'minizinc' no se encuentra, verifica PATH (which minizinc / where minizinc).
# - Puedes añadir/quitar solvers y estrategias en los flags correspondientes.