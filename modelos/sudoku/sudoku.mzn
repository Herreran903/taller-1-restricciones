% --------------------------------------------------------------
% sudoku.mzn — Sudoku 9x9 con all_different (+ sumas redundantes opcionales)
% - G[r,c] es la entrada: 0 = celda vacía, 1..9 = pista fija.
% - Restricciones principales:
%     * Filas, columnas y cajas 3x3 con all_different (propagación fuerte).
% - Restricciones redundantes (incluidas aquí como apoyo; puedes comentarlas):
%     * En cada fila/columna/caja: suma = 45.
%       (Las sumas no sustituyen a all_different; solo agregan chequeo/consistencia.)
% - Salida: tablero formateado con separadores de cajas.
% --------------------------------------------------------------

include "globals.mzn";

% ----------------------- Parámetros y conjuntos ----------------
int: N = 9;
set of int: S = 1..N;     % índices de filas/columnas
set of int: DIG = 1..N;   % dígitos válidos

% ----------------------- Datos (pistas) ------------------------
array[S,S] of 0..N: G;    % 0 = vacío, 1..9 = pista

% ----------------------- Variables -----------------------------
array[S,S] of var DIG: X;

% ----------------------- Fijar pistas --------------------------
constraint
  forall(r in S, c in S where G[r,c] > 0)(
    X[r,c] = G[r,c]
  );

% ----------------------- Restricciones -------------------------
% Filas: todos distintos
constraint
  forall(r in S)(
    all_different([ X[r,c] | c in S ])
  );

% Columnas: todos distintos
constraint
  forall(c in S)(
    all_different([ X[r,c] | r in S ])
  );

% Cajas 3x3: todos distintos
constraint
  forall(br in 0..2, bc in 0..2)(
    all_different([ X[3*br + i, 3*bc + j] | i in 1..3, j in 1..3 ])
  );

% ----------------------- Redundantes ---------------------------
% --- Redundantes por FILAS ---
% constraint
%   forall(r in S)(
%     sum(c in S)(X[r,c]) = 45
%   );
% %
% % --- Redundantes por COLUMNAS ---
% constraint
%   forall(c in S)(
%     sum(r in S)(X[r,c]) = 45
%   );
% %
% % --- Redundantes por CAJAS 3x3 ---
% constraint
%   forall(br in 0..2, bc in 0..2)(
%     sum(i in 1..3, j in 1..3)( X[3*br + i, 3*bc + j] ) = 45
%   );

% ----------------------- Búsqueda -------------------------------
% Ramificar solo sobre celdas vacías (las pistas ya están fijas)
array[int] of var DIG: BRANCH_VARS = [ X[r,c] | r in S, c in S where G[r,c] = 0 ];

% [H1] Heurística 1 — clásica y estable
% solve :: int_search(BRANCH_VARS, first_fail, indomain_min, complete) satisfy;

% [H2] Heurística 2 — informada por conflictos
% solve :: int_search(BRANCH_VARS, dom_w_deg, indomain_split, complete) satisfy;

% [H3] Heurística 3 — orden de entrada (baseline)
% solve :: int_search(BRANCH_VARS, input_order, indomain_min, complete) satisfy;

% [D] Default del solver (sin anotación)
solve satisfy;

% ----------------------- Salida --------------------------------
output
  [ "Solución Sudoku:\n","-------------------------\n" ]
  ++
  [ (if c = 1 then "| " else "" endif)
    ++ show(X[r,c])
    ++ (if c mod 3 = 0 then " | " else " " endif)
    ++ (if c = N then "\n" else "" endif)
    ++ (if c = N /\ r mod 3 = 0 then "-------------------------\n" else "" endif)
    | r in S, c in S
  ];
