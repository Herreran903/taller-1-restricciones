% --------------------------------------------------------------
% reunion.mzn — Ubicación de personas en una reunión (fila para foto)
% - Dado N personas y preferencias next/separate/distance, calcular un orden válido.
% - Restricciones principales:
%     * next(A,B): A y B deben estar juntos (adyacentes).
%     * separate(A,B): A y B no deben estar juntos (no adyacentes).
%     * distance(A,B,M): a lo sumo M personas entre A y B.
% - Búsqueda: SIN estrategia fija (usa la búsqueda por defecto del solver).
% - Salida: orden formateado con nombres y posiciones.
% --------------------------------------------------------------

include "globals.mzn";

% ----------------------- Parámetros y conjuntos ----------------
int: N;                   % número de personas
set of int: S = 1..N;     % índice de personas
set of int: POS = 1..N;   % dominio de posiciones

% ----------------------- Datos (preferencias) ------------------
% Nombres
array[S] of string: personas;

% Cantidades por tipo de preferencia
int: K_NEXT;
int: K_SEP;
int: K_DIST;

% Matrices de preferencias (índices 1..K_*, columnas fijas):
% NEXT[i,1]=A, NEXT[i,2]=B
% SEP[i,1]=A,  SEP[i,2]=B
% DIST[i,1]=A, DIST[i,2]=B, DIST[i,3]=M
array[1..K_NEXT, 1..2] of int: NEXT;
array[1..K_SEP,  1..2] of int: SEP;
array[1..K_DIST, 1..3] of int: DIST;

% ----------------------- Variables -----------------------------
% POS_OF[p] = posición (1..N) que ocupa la persona p
% PER_AT[i] = persona ubicada en la posición i
array[S]   of var POS: POS_OF;
array[POS] of var S:   PER_AT;

% ----------------------- Relación de permutación ---------------
% Canalización biyección: garantiza que es una permutación válida
constraint inverse(POS_OF, PER_AT);

% ----------------------- Restricciones -------------------------
% next(A,B): adyacentes → |POS_OF[A] - POS_OF[B]| = 1
constraint
  forall(i in 1..K_NEXT)(
    abs(POS_OF[NEXT[i,1]] - POS_OF[NEXT[i,2]]) = 1
  );

% separate(A,B): no adyacentes → |POS_OF[A] - POS_OF[B]| >= 2
constraint
  forall(i in 1..K_SEP)(
    abs(POS_OF[SEP[i,1]] - POS_OF[SEP[i,2]]) >= 2
  );

% distance(A,B,M): a lo sumo M personas entre A y B ⇒ |POS_OF[A] - POS_OF[B]| <= M + 1
constraint
  forall(i in 1..K_DIST)(
    abs(POS_OF[DIST[i,1]] - POS_OF[DIST[i,2]]) <= DIST[i,3] + 1
  );

% ----------------------- Redundantes ---------------------------
% Evita que una persona aparezca en demasiadas relaciones NEXT
constraint forall(p in S)(
  sum([ bool2int(p == NEXT[i,1] \/ p == NEXT[i,2]) | i in 1..K_NEXT ]) <= 2
);

% Evita pares que sean a la vez NEXT y SEPARATE
constraint
forall(i in 1..K_NEXT, j in 1..K_SEP)(
  not (
    (NEXT[i,1] == SEP[j,1] /\ NEXT[i,2] == SEP[j,2]) \/
    (NEXT[i,1] == SEP[j,2] /\ NEXT[i,2] == SEP[j,1])
  )
);

% ----------------------- Simetría ------------
constraint PER_AT[1] < PER_AT[N];

% ----------------------- Búsqueda -------------------------------
% [H1] Heurística 1 — clásica y estable
% solve :: int_search(POS_OF, first_fail, indomain_min, complete) satisfy;

% [H2] Heurística 2 — informada por conflictos
solve :: int_search(POS_OF, dom_w_deg, indomain_split, complete) satisfy;

% [D] Default del solver (sin anotación)
solve satisfy;

% ----------------------- Salida --------------------------------
output
  [ "Orden de la fila (1.." ++ show(N) ++ "):\n" ]
  ++
  [ show(i) ++ ". " ++ personas[fix(PER_AT[i])] ++ "\n" | i in POS ];
