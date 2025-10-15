% -------------------------------------------
% acertijo.mzn — Atributos como enteros + salida legible
% -------------------------------------------
include "globals.mzn";  % para alldifferent

% ----------------------- Parámetros ----------------
enum NOMBRE = {Juan, Oscar, Dario};
set of int: AGE = {24,25,26};

% Índices (constantes)
int: GONZALEZ = 1; int: GARCIA = 2; int: LOPEZ = 3;
int: CLASICA  = 1; int: POP    = 2; int: JAZZ  = 3;

% ----------------------- Variables -----------------------------
array[NOMBRE] of var 1..3: apellido;   % 1..3
array[NOMBRE] of var 1..3: musica;     % 1..3
array[NOMBRE] of var AGE:  edad;          % 24,25,26

% “punteros”
var NOMBRE: p_gonzalez;
var NOMBRE: p_pop;

% ----------------------- Restricciones -------------------------
constraint alldifferent([apellido[n] | n in NOMBRE]);
constraint alldifferent([musica[n]   | n in NOMBRE]);
constraint alldifferent([edad[n]        | n in NOMBRE]);

% Enlaces de punteros
constraint apellido[p_gonzalez] = GONZALEZ;
constraint musica[p_pop] = POP;

% Pistas del acertijo
constraint edad[Juan] > edad[p_gonzalez];
constraint musica[p_gonzalez] = CLASICA;

constraint apellido[p_pop] != GARCIA;
constraint edad[p_pop] != 24;

constraint apellido[Oscar] != LOPEZ;
constraint edad[Oscar] = 25;

constraint musica[Dario] != JAZZ;

% ----------------------- Búsqueda -------------------------------
solve :: int_search(
          [apellido[n] | n in NOMBRE] ++
          [musica[n]   | n in NOMBRE] ++
          [edad[n]        | n in NOMBRE],
          first_fail, indomain_min, complete
       ) satisfy;

% ----------------------- Salida -------------------------------
array[1..3] of string: ap_str = ["Gonzalez","Garcia","Lopez"];
array[1..3] of string: mu_str = ["Clasica","Pop","Jazz"];

output
  ["SOLUCIÓN:\n"] ++
  [
    let {
      int: ai = fix(apellido[n]),
      int: mi = fix(musica[n]),
      int: ei = fix(edad[n])
    } in
    show(n) ++ " " ++ ap_str[ai] ++ ", " ++ show(ei) ++ ", " ++ mu_str[mi] ++ "\n"
    | n in NOMBRE
  ];