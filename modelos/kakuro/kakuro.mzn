% --------------------------------------------------------------
% kakuro.mzn — Modelo de Kakuro con restricciones de suma y unicidad
% - Cada casilla blanca contiene un número del 1 al 9.
% - Las pistas horizontales y verticales indican la suma de los dígitos
%   consecutivos en esa dirección (sin repetición de dígitos).
% - Restricciones principales:
%     * Sumas horizontales y verticales deben coincidir con las pistas dadas.
%     * Dentro de cada grupo de casillas (across/down) los dígitos son distintos.
% - Restricciones de consistencia:
%     * Validación de coordenadas y longitudes de pistas.
%     * Casillas negras fijadas con valor constante.
% - Restricciones redundantes:
%     * La suma de las casillas de una pista deben ser mayores que el valor resultante de llenar la
%       pista solo con 1 y menores que llenar la pista solo con 9.
% - Salida: tablero con valores asignados en las casillas blancas y X en casillas negras.
% --------------------------------------------------------------

include "globals.mzn";

% ----------------------- Parámetros y conjuntos ----------------
int: R; % Filas
int: C; % Columnas
set of int: Rows = 1..R;
set of int: Cols = 1..C;

array[Cols,Rows] of int: white; % Matriz de casillas: 0 = Negra, 1 = Blanca

int: NA; % Pistas horizontales
array[1..NA] of int: a_row; % Fila donde inicia la pista
array[1..NA] of int: a_col; % Columna donde se sitúa la pista
array[1..NA] of int: a_len; % Longitud de la pista
array[1..NA] of int: a_sum; % Suma de la pista

int: ND; % Pistas verticales
array[1..ND] of int: d_row; % Fila donde se sitúa la pista
array[1..ND] of int: d_col; % Columna donde inicia la pista
array[1..ND] of int: d_len; % Longitud de la pista
array[1..ND] of int: d_sum; % Suma de la pista

% ----------------------- Variables -----------------------------
array[Cols,Rows] of var 1..9: x;

% ----------------------- Restricciones -------------------------

% Fijar casillas negras
constraint
  forall(c in Cols, r in Rows where white[c,r] = 0) (
    x[c,r] = 1
  );

% Restricción pistas dentro de casillas blancas
constraint
  forall(i in 1..NA) (
    a_row[i] >= 1 /\ a_row[i] <= R /\
    a_col[i] >= 1 /\ a_col[i] <= C /\
    a_row[i] + a_len[i] - 1 <= R /\
    forall(k in 0..a_len[i]-1)( white[a_col[i], a_row[i]+k] = 1 )
  );

constraint
  forall(i in 1..ND) (
    d_col[i] >= 1 /\ d_col[i] <= C /\
    d_row[i] >= 1 /\ d_row[i] <= R /\
    d_col[i] + d_len[i] - 1 <= C /\
    forall(k in 0..d_len[i]-1)( white[d_col[i]+k, d_row[i]] = 1 )
  );
  
% Restricciones para pistas horizontales
constraint
  forall(i in 1..NA) (
    % Restricción de suma
      sum([ x[a_col[i], a_row[i] + k] | k in 0..a_len[i]-1 ]) = a_sum[i] /\
    % Todos diferentes
      all_different([ x[a_col[i], a_row[i] + k] | k in 0..a_len[i]-1 ])
  );
  
% Restricciones para pistas verticales
constraint
  forall(i in 1..ND) (
    % Restricción de suma
      sum([ x[d_col[i] + k, d_row[i]] | k in 0..d_len[i]-1 ]) = d_sum[i] /\
    % Todos diferentes
      all_different([ x[d_col[i] + k, d_row[i]] | k in 0..d_len[i]-1 ])
  );
 
  
% ----------------------- Redundantes ---------------------------

% --- Redundante suma horizontal ---
%constraint
%  forall(i in 1..NA) (
%    sum([ x[a_col[i], a_row[i] + k] | k in 0..a_len[i]-1 ]) <= a_len[i] * 9 /\
%    sum([ x[a_col[i], a_row[i] + k] | k in 0..a_len[i]-1 ]) >= a_len[i] * 1 
%  );

% --- Redundante suma vertical ---
%constraint
%  forall(i in 1..ND) (
%    sum([ x[d_col[i] + k, d_row[i]] | k in 0..d_len[i]-1 ]) <= d_len[i] * 9 /\
%    sum([ x[d_col[i] + k, d_row[i]] | k in 0..d_len[i]-1 ]) >= d_len[i] * 1 
%  );

% ----------------------- Búsqueda -------------------------------

% Ramificar solo sobre celdas blancas
array[int] of var 1..9: SEARCH_VARS =
  [ x[c,r] | c in Cols, r in Rows where white[c,r] = 1 ];

% [H1] Heurística 1 — clásica y estable
% solve :: int_search(SEARCH_VARS, first_fail, indomain_min, complete) satisfy;

% [H2] Heurística 2 — informada por conflictos
solve :: int_search(SEARCH_VARS, dom_w_deg, indomain_split, complete) satisfy;

% [D] Default del solver (sin anotación)
%solve satisfy;

% ----------------------- Salida --------------------------------
output [
  if r=1 then "\n" else " " endif ++
  if white[c,r]=0 then "X" else show(x[c,r]) endif
  | c in Cols, r in Rows
];