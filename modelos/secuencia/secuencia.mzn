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

% ----------------------- Variables de decisión -------------------------------

array[int] of var int: DECISION_VARS = [ x[i] | i in 0..n-1 ];

% ----------------------- Búsqueda -------------------------------

% solve satisfy;

% first_fail + indomain_min: elige la var con dominio más pequeño, prueba el menor valor y hace DFS completa (poda temprana).
% solve :: int_search(x, first_fail, indomain_min, complete) satisfy;

% input_order + indomain_split: respeta el orden de x tal cual; en cada variable parte el dominio (mitad inferior primero) para forzar propagación; búsqueda completa (DFS).
 solve :: int_search(x, input_order, indomain_split, complete) satisfy;

% input_order + indomain_min: sigue el orden de x tal cual, prueba el menor valor y hace DFS completa (control/reproducible).
%  solve :: int_search(x, input_order, indomain_min, complete) satisfy;
% ----------------------- Salida -------------------------------
output ["x = ", show(x), "\n"];