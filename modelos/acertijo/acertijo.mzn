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

array[int] of var int: vars =
    [ apellido[n] | n in NOMBRE ]
 ++ [ musica[n]   | n in NOMBRE ]
 ++ [ edad[n]     | n in NOMBRE ];

% ----------------------- Búsqueda -------------------------------
% solve satisfy;

% first_fail + indomain_min: elige la var con dominio más pequeño, prueba el menor valor y hace DFS completa (poda temprana).
% solve :: int_search(vars, first_fail, indomain_min, complete) satisfy;

% input_order + indomain_split: respeta el orden de x tal cual; en cada variable parte el dominio (mitad inferior primero) para forzar propagación; búsqueda completa (DFS).
% solve :: int_search(vars, input_order, indomain_split, complete) satisfy;

% input_order + indomain_min: sigue el orden de x tal cual, prueba el menor valor y hace DFS completa (control/reproducible).
 solve :: int_search(vars, input_order, indomain_min, complete) satisfy;

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