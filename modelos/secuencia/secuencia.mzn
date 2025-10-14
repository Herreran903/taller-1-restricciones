% -------------------------------------------
% secuencia.mzn — Secuencias mágicas de longitud n
% -------------------------------------------

int: n;                                  % parámetro

array[0..n-1] of var 0..n-1: x;          % cada x[i] en 0..n-1

% Definición: el número i aparece exactamente x[i] veces
constraint
  forall(i in 0..n-1)(
    x[i] = sum(j in 0..n-1)( bool2int(x[j] = i) )
  );

% Redundantes que podan búsqueda (recomendadas)
constraint sum(i in 0..n-1)(x[i]) = n;
constraint sum(i in 0..n-1)((i-1) * x[i]) = 0;

solve satisfy;

output ["x = ", show(x), "\n"];