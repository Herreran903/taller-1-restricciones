% rectangle.mzn
include "globals.mzn";

% ----------------------- Parámetros y conjuntos ----------------
int: n; % Número de cuadrados
array[1..n] of int: s; % Tamaño de los cuadrados
int: W; % Ancho del rectángulo
int: H; % Alto del rectángulo

% ----------------------- Variables -----------------------------
array[1..n] of var 0..W: x;
array[1..n] of var 0..H: y;

% ----------------------- Restricciones -------------------------

% Restricción cuadrados dentro del rectángulo
constraint
  forall(i in 1..n) (
    x[i] + s[i] <= W /\
    y[i] + s[i] <= H
  );

% Restricción cuadrados no se solapan
constraint diffn(x, y, s, s);

% ----------------------- Redundantes ---------------------------

% --- Redundante suma de área ---
constraint sum(i in 1..n)(s[i]*s[i]) <= W * H;

% ----------------------- Simetría ------------
constraint forall(i, j in 1..n where i < j /\ s[i] = s[j]) (
    lex_lesseq([x[i], y[i]], [x[j], y[j]])
);

% ----------------------- Búsqueda -------------------------------

% Ramificar sobre x, y
array[1..(2*n)] of var int: SEARCH_VARS =
  [ x[i] | i in 1..n ] ++ [ y[i] | i in 1..n ];

% [H1] Heurística 1 — clásica y estable
% solve :: int_search(SEARCH_VARS, first_fail, indomain_min, complete) satisfy;

% [H2] Heurística 2 — informada por conflictos
solve :: int_search(SEARCH_VARS, dom_w_deg, indomain_split, complete) satisfy;

% [D] Default del solver (sin anotación)
% solve satisfy;

% ----------------------- Salida --------------------------------
output [
  "Rectángulo: ", show(W), "x", show(H), "\n",
  "Cuadrados: ", show(s), "\n",
  "Coordenadas:\n"
] ++
let {
  array[1..n] of string: lineas =
    [ "  Cuadrado " ++ show(i) ++
      " (lado=" ++ show(s[i]) ++
      ") -> (x=" ++ show(x[i]) ++
      ", y=" ++ show(y[i]) ++ ")\n"
    | i in 1..n ]
} in
[ join("", lineas) ];