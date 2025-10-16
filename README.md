# Taller 1 — Modelamiento e Implementación de CSPs en MiniZinc

**Equipo:** John Freddy Belalcazar · Samuel Galindo Cuevas · Nicolás Herrera Marulanda  
**Asignatura:** Programación por Restricciones — Escuela de Ingeniería de Sistemas y Computación, Universidad del Valle  
**Profesor:** 
Robinson Andrey Duque Agudelo  
**Fecha:** Octubre de 2025  

---

## Descripción general

Este repositorio contiene la implementación de los **seis problemas propuestos** en el Taller 1 de *Programación por Restricciones*.  
Cada problema se formula como un **problema de satisfacción de restricciones (CSP)** y se implementa en **MiniZinc**, explorando diferentes estrategias de búsqueda, restricciones redundantes y mecanismos de ruptura de simetría.

El repositorio incluye además un **informe en LaTeX** con la estructura oficial (modelo, implementación, pruebas, árboles de búsqueda, análisis y conclusiones), y un **script auxiliar en Python** para automatizar la ejecución de los modelos sobre múltiples instancias y solvers.

---

## Estructura del repositorio

```
taller-1/
├─ main.tex
├─ preambulo.tex
├─ README.md
├─ .gitignore  .gitattributes  .editorconfig  Makefile
│
├─ refs/
│  ├─ john.bib
│  ├─ samuel.bib
│  └─ nicolas.bib
│
├─ secciones/
│  ├─ 01-sudoku/
│  │  ├─ 00-sudoku-intro.tex
│  │  ├─ 01-modelo.tex
│  │  ├─ 02-implementacion.tex
│  │  ├─ 03-pruebas.tex
│  │  ├─ 04-arboles.tex
│  │  └─ 05-analisis-y-conclusiones.tex
│
├─ modelos/
│  ├─ sudoku/
│  │  ├─ sudoku.mzn
│  │  ├─ tests/           # Instancias .dzn
│  │  └─ resultados/      # Salidas o estadísticas
│  ├─ kakuro/
│  │  ├─ kakuro.mzn
│  │  ├─ tests/
│  │  └─ resultados/
│  ├─ secuencia/
│  │  ├─ secuencia.mzn
│  │  ├─ tests/
│  │  └─ resultados/
│  ├─ acertijo/
│  │  ├─ acertijo.mzn
│  │  ├─ tests/
│  │  └─ resultados/
│  ├─ reunion/
│  │  ├─ reunion.mzn
│  │  ├─ tests/
│  │  └─ resultados/
│  └─ rectangulo/
│     ├─ rectangulo.mzn
│     ├─ tests/
│     └─ resultados/
│
└─ script/
   └─ run_experiments.py   # Barrido automático de MiniZinc
```

> **Nota:** dentro de cada carpeta de `modelos/<problema>/` se encuentran:
> - El modelo principal (`.mzn`).
> - Las instancias de prueba (`tests/*.dzn`).
> - Una carpeta opcional `resultados/` con salidas o estadísticas de ejecución.

---

## Ejecución de los modelos

Los modelos pueden ejecutarse de dos maneras:

### 1. Desde la aplicación MiniZinc IDE

1. Abrir el archivo `.mzn` correspondiente (por ejemplo `modelos/sudoku/sudoku.mzn`).
2. Asociar un archivo `.dzn` desde la pestaña *Data*.
3. Seleccionar el solver (por ejemplo `Gecode` o `Chuffed`).
4. Ejecutar (`Ctrl+R` o botón *Run*).
5. Revisar el árbol de búsqueda y estadísticas.

### 2. Desde la consola (CLI)

**macOS / Linux**
```bash
minizinc --solver Gecode modelos/sudoku/sudoku.mzn modelos/sudoku/tests/test_01.dzn
```

**Windows (PowerShell)**
```powershell
minizinc --solver Gecode modelos\sudoku\sudoku.mzn modelos\sudoku\tests\test_01.dzn
```

> También puede añadirse la opción `--statistics` para obtener datos de búsqueda y fallos:
> ```bash
> minizinc --solver Chuffed --statistics modelos/reunion/reunion.mzn modelos/reunion/tests/test_02.dzn
> ```

---

## Ejecución automatizada con el script

El archivo `script/run_experiments.py` automatiza la ejecución masiva de modelos sobre múltiples instancias, estrategias y solvers.

### Requisitos

- **MiniZinc** instalado y disponible en el `PATH`.
- **Python 3.x**.

#### Añadir MiniZinc al `PATH`

**macOS (MiniZinc IDE)**
```bash
export PATH="/Applications/MiniZincIDE.app/Contents/Resources:$PATH"
```

**Windows (PowerShell)**
```powershell
$env:PATH = "C:\Program Files\MiniZinc\;" + $env:PATH
```

**Linux**
Depende del gestor de paquetes (por ejemplo `apt install minizinc`); normalmente se añade automáticamente.

### Ejemplo general

```bash
python3 script/run_experiments.py   --base-dir modelos/sudoku   --model    sudoku.mzn   --data-dir tests   --solver   gecode chuffed   --strategy ff_min wdeg_split inorder_min   --time-limit 60000
```

### Estrategias disponibles

- `ff_min` → `first_fail` + `indomain_min`
- `wdeg_split` → `dom_w_deg` + `indomain_split`
- `inorder_min` → `input_order` + `indomain_min`

> El script reconoce automáticamente los modelos “Sudoku” y “Reunión” para inyectar la lista de variables de decisión y la anotación `int_search(...)` antes del `solve`.

---

## Ejemplos por problema

### Sudoku
```bash
python3 script/run_experiments.py   --base-dir modelos/sudoku   --model    sudoku.mzn   --data-dir tests   --solver   gecode   --strategy ff_min   --time-limit 60000
```

### Kakuro
```bash
python3 script/run_experiments.py   --base-dir modelos/kakuro   --model    kakuro.mzn   --data-dir tests   --solver   gecode   --strategy inorder_min
```

### Secuencia Mágica
```bash
python3 script/run_experiments.py   --base-dir modelos/secuencia   --model    secuencia.mzn   --data-dir tests   --solver   gecode   --strategy wdeg_split
```

### Acertijo Lógico
```bash
python3 script/run_experiments.py   --base-dir modelos/acertijo   --model    acertijo.mzn   --data-dir tests   --solver   gecode
```

### Reunión
```bash
python3 script/run_experiments.py   --base-dir modelos/reunion   --model    reunion.mzn   --data-dir tests   --solver   chuffed   --strategy ff_min
```

### Rectángulo
```bash
python3 script/run_experiments.py   --base-dir modelos/rectangulo   --model    rectangulo.mzn   --data-dir tests   --solver   gecode
```