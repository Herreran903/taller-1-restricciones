# Barrido de pruebas para un modelo MiniZinc (genérico)

**Requisitos:** MiniZinc en `PATH` y Python 3.x  
**Modelo:** `<ruta/al/modelo.mzn>` (p. ej., `modelos/<PROBLEMA>/<modelo>.mzn`)  
**Instancias:** `<ruta/a/tests>/*.dzn` (p. ej., `modelos/<PROBLEMA>/tests`)

> 💡 El script detecta automáticamente el tipo de modelo:
> - **Sudoku** (matriz `X` con pistas `G`)
> - **Reunión** (permuta con `POS_OF`/`PER_AT` e `inverse`)
>
> Si hace falta, inyecta una lista de ramificación adecuada antes del `solve`
> y sustituye la línea `solve ... satisfy;` por la estrategia pedida.
> Si no reconoce el modelo, ejecuta el `solve` por defecto (sin estrategia).

---

## macOS / Linux

**Asegurar MiniZinc en PATH:**
```bash
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
```

**Barrido genérico (multi-solver / multi-estrategia):**
```bash
python3 script/run_experiments.py \
  --base-dir modelos/<PROBLEMA> \
  --model    modelos/<PROBLEMA>/<modelo>.mzn \
  --data-dir modelos/<PROBLEMA>/tests \
  --solver   chuffed gecode \
  --strategy ff_min domdeg_split input_min default \
  --time-limit 60000
```

---

## Windows (PowerShell)

**Asegurar MiniZinc en PATH para la sesión actual:**
```powershell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
```

**Barrido genérico:**
```powershell
python script\run_experiments.py `
  --base-dir modelos\<PROBLEMA> `
  --model    modelos\<PROBLEMA>\<modelo>.mzn `
  --data-dir modelos\<PROBLEMA>\tests `
  --solver   chuffed gecode `
  --strategy ff_min domdeg_split input_min default `
  --time-limit 60000
```

**Alternativa en CMD:**
```bat
rem set "PATH=C:\Program Files\MiniZinc;%PATH%"
python script\run_experiments.py ^
  --base-dir modelos\<PROBLEMA> ^
  --model    modelos\<PROBLEMA>\<modelo>.mzn ^
  --data-dir modelos\<PROBLEMA>\tests ^
  --solver   chuffed gecode ^
  --strategy ff_min domdeg_split input_min default ^
  --time-limit 60000
```

---

## Ejemplos

### Sudoku
```bash
# macOS / Linux
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
python3 script/run_experiments.py \
  --base-dir modelos/sudoku \
  --model    modelos/sudoku/sudoku.mzn \
  --data-dir modelos/sudoku/tests \
  --solver   gecode \
  --strategy ff_min \
  --time-limit 60000
```

```powershell
# Windows PowerShell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
python script\run_experiments.py `
  --base-dir modelos\sudoku `
  --model    modelos\sudoku\sudoku.mzn `
  --data-dir modelos\sudoku\tests `
  --solver   gecode `
  --strategy ff_min `
  --time-limit 60000
```

### Reunión (fila para foto)
```bash
# macOS / Linux
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
python3 script/run_experiments.py \
  --base-dir modelos/reunion \
  --model    modelos/reunion/reunion.mzn \
  --data-dir modelos/reunion/tests \
  --solver   gecode \
  --strategy ff_min \
  --time-limit 60000
```

```powershell
# Windows PowerShell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
python script\run_experiments.py `
  --base-dir modelos\reunion `
  --model    modelos\reunion\reunion.mzn `
  --data-dir modelos\reunion\tests `
  --solver   gecode `
  --strategy ff_min `
  --time-limit 60000
```

---

## Salidas (todas relativas a `--base-dir`)

- `results/results.csv` — todas las corridas (incluye `time_raw` y `time`).
- `results/artifacts/` — logs y soluciones de **cada** corrida:
  - `*.log.txt` — comando, stdout y stderr.
  - `*.sol.txt` — salida de soluciones del solver.
- `results/shortlist.csv` — corridas “interesantes” (mejores/peores por instancia, Pareto, outliers, gaps, anomalías).
- `results/shortlist_artifacts/` — copia de `*.log.txt` y `*.sol.txt` solo para la shortlist.

> ℹ️ El script reintenta automáticamente con `solve` por defecto si detecta un
> identificador indefinido (p. ej., lista de branching) o si el modelo no es reconocible.

---

## Notas

- `rc=0` solo indica que el proceso ejecutó; revisa `status` para `SAT/UNSAT/UNKNOWN`.
- Si `minizinc` no se encuentra, verifica el PATH (`which minizinc` / `where minizinc`).
- Puedes etiquetar el modelo para detección determinista añadiendo al inicio:
  - En `sudoku.mzn`:  
    `% MODEL_ID: sudoku`
  - En `reunion.mzn`:  
    `% MODEL_ID: reunion`
