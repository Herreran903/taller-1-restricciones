# Taller 1 — Informe LaTeX + Experimentos MiniZinc

**Equipo:** John Freddy Belalcazar · Samuel Galindo Cuevas · Nicolas Herrera Marulanda

Este repositorio está organizado **por problema** y cada problema se divide en las secciones del informe (*Intro, Modelo, Implementación, Pruebas, Árboles, Análisis y conclusiones*).  
Además, incluye un script para barrer instancias **MiniZinc** con distintas **estrategias de búsqueda** y **solvers**.

---

## Estructura

```
taller-1/
├─ main.tex
├─ preambulo.tex
├─ README.md
├─ .gitignore  .gitattributes  .editorconfig  Makefile
├─ refs/
│  ├─ john.bib  ├─ samuel.bib  └─ nicolas.bib
├─ secciones/
│  └─ 01-sudoku/
│     ├─ 00-sudoku-intro.tex
│     ├─ 01-modelo.tex
│     ├─ 02-implementacion.tex
│     ├─ 03-pruebas.tex
│     ├─ 04-arboles.tex
│     └─ 05-analisis-y-conclusiones.tex
├─ modelos/
│  └─ sudoku/
│     ├─ sudoku.mzn
│     └─ tests/            # .dzn
└─ script/
   └─ run_experiments.py   # barrido de MiniZinc
```

> **Nota:** en la sección de **modelos** no se documenta aquí ninguna carpeta de resultados.

---

## Cómo colaborar (LaTeX)

- Trabaja en tu sección/archivo dentro de `secciones/01-sudoku/`.
- Usa **una oración por línea** en `.tex` para facilitar los diffs.
- Cambios globales (`preambulo.tex`, `main.tex`) → **PRs pequeños**.
- Bibliografía: añade tus entradas a `refs/*.bib` y cita con `\citep{clave}` o `\citet{clave}`.

### Compilar el informe

**Requisito:** TeX Live / MacTeX.

```bash
latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex
# Alternativa (ejecuta 2–3 veces si hay referencias cruzadas):
pdflatex main.tex
```

---

## Experimentos MiniZinc

El script `script/run_experiments.py` compila y ejecuta un modelo `.mzn` sobre todas las instancias `.dzn` de una carpeta, probando una o varias **estrategias** y **solvers**.

### Requisitos

- MiniZinc instalado y disponible en `PATH`.
- Python 3.x.

#### Poner MiniZinc en el `PATH`

**macOS (MiniZinc IDE):**
```bash
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
```

**Windows (PowerShell):**
```powershell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
```

**Linux:** depende del gestor de paquetes; tras instalar, suele bastar.

---

## Estrategias de búsqueda disponibles

- `ff_min` → `first_fail` + `indomain_min` (**completa**)
- `wdeg_split` → `dom_w_deg` + `indomain_split` (**completa**)  
  *(alias aceptado: `domdeg_split`)*
- `inorder_min` → `input_order` + `indomain_min` (**completa**)

> El script intenta inyectar `solve :: int_search(...) satisfy;` y una lista de variables de ramificación si el modelo lo permite (detecta Sudoku o Reunión).  
> Si no reconoce el modelo, ejecuta el `solve` original del `.mzn`.

---

## Uso genérico

**macOS / Linux**
```bash
python3 script/run_experiments.py \
  --base-dir modelos/sudoku \
  --model    sudoku.mzn \
  --data-dir tests \
  --solver   chuffed gecode \
  --strategy ff_min wdeg_split inorder_min \
  --time-limit 60000
```

**Windows (PowerShell)**
```powershell
python script\run_experiments.py `
  --base-dir modelos\sudoku `
  --model    sudoku.mzn `
  --data-dir tests `
  --solver   chuffed gecode `
  --strategy ff_min wdeg_split inorder_min `
  --time-limit 60000
```

---

## Ejemplos por problema

### Sudoku
```bash
# macOS / Linux
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
python3 script/run_experiments.py \
  --base-dir modelos/sudoku \
  --model    sudoku.mzn \
  --data-dir tests \
  --solver   gecode \
  --strategy ff_min \
  --time-limit 60000
```

### Reunión (si existe `modelos/reunion/`)
```bash
# macOS / Linux
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
python3 script/run_experiments.py \
  --base-dir modelos/reunion \
  --model    reunion.mzn \
  --data-dir tests \
  --solver   gecode \
  --strategy ff_min \
  --time-limit 60000
```

---

## Parámetros principales

- `--base-dir` — Carpeta base (todas las rutas relativas parten de aquí).
- `--model` — Ruta al `.mzn` (relativa a `--base-dir` si no es absoluta).
- `--data-dir` — Carpeta con `.dzn` (relativa a `--base-dir`).
- `--solver` — Uno o varios: `gecode`, `chuffed` (o el que tengas instalado).
- `--strategy` — Una o varias: `ff_min`, `wdeg_split` (`domdeg_split`), `inorder_min`.
- `--time-limit` — Tiempo máximo por corrida (ms).

---

## Consejos rápidos

- Verifica MiniZinc: `which minizinc` (macOS/Linux) o `where minizinc` (Windows).
- Si tu modelo es Sudoku, puedes etiquetarlo para detección determinista al inicio del `.mzn`:

```minizinc
% MODEL_ID: sudoku
```

*(Análogo para Reunión: `MODEL_ID: reunion`.)*

- Si el script avisa que **no puede inyectar** la estrategia, ejecutará el `solve` del modelo tal cual.

---

## Licencia y créditos

Código y modelos para uso académico en el marco del taller.  
Citas y bibliografía en `refs/*.bib`.
