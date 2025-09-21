# Taller 1 — LaTeX colaborativo (Git)

**Equipo:** John Freddy Belalcazar, Samuel Galindo Cuevas, Nicolas Herrera Marulanda

Este repositorio está organizado **por problema** (según el documento del taller) y, dentro de cada problema, por los apartados exigidos en el informe: *Modelo, Detalles de implementación, Árboles de búsqueda, Pruebas, Análisis, Conclusiones*.

## Estructura
```
taller-1/
├─ main.tex
├─ preambulo.tex
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ .editorconfig
├─ Makefile
├─ refs/
│  ├─ john.bib
│  ├─ samuel.bib
│  └─ nicolas.bib
├─ secciones/
│  ├─ 01-sudoku/
│  │  ├─ 00-sudoku-intro.tex
│  │  ├─ 01-modelo.tex
│  │  ├─ 02-implementacion.tex
│  │  ├─ 03-arboles.tex
│  │  ├─ 04-pruebas.tex
│  │  ├─ 05-analisis.tex
│  │  └─ 06-conclusiones.tex
│  ├─ 02-kakuro/ (… mismo patrón …)
│  ├─ 03-secuencia-magica/ (…)
│  ├─ 04-acertijo-logico/ (…)
│  ├─ 05-reunion/ (…)
│  └─ 06-rectangulo/ (…)
├─ figuras/
└─ tablas/
```

## Cómo colaborar sin conflictos
- Cada persona trabaja en **sus archivos** dentro del problema asignado.
- Mantén la convención de **una oración por línea** en `.tex`.
- Cambios globales (`preambulo.tex`, `main.tex`) van en PRs pequeños.
- Bibliografía: agrega tus entradas a `refs/john.bib`, `refs/samuel.bib` o `refs/nicolas.bib` y cita con `\citep{clave}` o `\citet{clave}`.

## Compilación
- **Local (recomendado):** `latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex`
  - Requiere **TeX Live / MacTeX** (en macOS: `brew install --cask mactex-no-gui`).
  - Alternativa manual: `pdflatex main.tex` (ejecútalo 2–3 veces si necesitas referencias cruzadas).
- **CI (GitHub Actions):** usando **latexmk/pdflatex** con una acción de LaTeX (ver .github/workflows/latex.yml).

