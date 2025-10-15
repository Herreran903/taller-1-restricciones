% -------------------------------------------
% secuencia.mzn — Secuencias mágicas de longitud n
% -------------------------------------------

% ----------------------- Parámetros ----------------

int: n;                                  

% ----------------------- Variables -----------------------------

array[0..n-1] of var 0..n-1: x;        

% ----------------------- Restricciones -------------------------

% Definición: el número i aparece exactamente x[i] veces
constraint
  forall(i in 0..n-1)(
    count(x, i) = x[i]   
  );

% Redundantes que podan búsqueda
constraint sum(i in 0..n-1)(x[i]) = n;
constraint sum(i in 0..n-1)((i-1) * x[i]) = 0;

% ----------------------- Búsqueda -------------------------------

% solve satisfy;

% Clásico: variables con dominio más pequeño primero, elige el menor valor
%solve :: int_search(x, first_fail, indomain_min, complete) satisfy;

% A veces ayuda lo contrario (deja los difíciles para después)
% solve :: int_search(x, anti_first_fail, indomain_min, complete) satisfy;

% Partición del dominio (branch & bound estilo split)
% solve :: int_search(x, first_fail, indomain_split, complete) satisfy;


% ----------------------- Salida -------------------------------
output ["x = ", show(x), "\n"];