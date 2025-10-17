% -------------------------------------------
% acertijo_sin_punteros.mzn — Sin punteros
% -------------------------------------------
include "globals.mzn";

% ----------------------- Parámetros ----------------
enum NOMBRE = {Juan, Oscar, Dario};
set of int: AGE = {24,25,26};

% Códigos para atributos
int: GONZALEZ = 1; int: GARCIA = 2; int: LOPEZ = 3;
int: CLASICA  = 1; int: POP    = 2; int: JAZZ  = 3;

% ----------------------- Variables -----------------
array[NOMBRE] of var 1..3: apellido;   % 1..3
array[NOMBRE] of var 1..3: musica;     % 1..3
array[NOMBRE] of var AGE:  edad;       % 24,25,26

% ----------------------- Restricciones --------------
% Biyectividad de asignaciones
constraint alldifferent([apellido[n] | n in NOMBRE]);
constraint alldifferent([musica[n]   | n in NOMBRE]);
constraint alldifferent([edad[n]     | n in NOMBRE]);


% (1) Juan es mayor que la persona de apellido Gonzalez
constraint forall(n in NOMBRE)(
  (apellido[n] = GONZALEZ) -> (edad[Juan] > edad[n])
);

% (2) La persona de apellido Gonzalez escucha clásica
constraint forall(n in NOMBRE)(
  (apellido[n] = GONZALEZ) -> (musica[n] = CLASICA)
);

% (3) Quien escucha Pop no es García
constraint forall(n in NOMBRE)(
  (musica[n] = POP) -> (apellido[n] != GARCIA)
);

% (4) Quien escucha Pop no tiene 24 años
constraint forall(n in NOMBRE)(
  (musica[n] = POP) -> (edad[n] != 24)
);

% --- Resto de pistas ---
constraint apellido[Oscar] != LOPEZ;
constraint edad[Oscar] = 25;
constraint musica[Dario] != JAZZ;

% ----------------------- Redundantes útiles --------
% Cada valor aparece exactamente una vez (ya lo implica alldifferent + dominios),
% pero esto suele podar antes.
% constraint count([apellido[n] | n in NOMBRE], GONZALEZ) = 1;
% constraint count([musica[n]   | n in NOMBRE], POP)      = 1;

% % La suma de las edades es fija (24+25+26=75), consecuencia de alldifferent(edad).
% constraint sum(n in NOMBRE)(edad[n]) = 24 + 25 + 26;

% ----------------------- Búsqueda -------------------
array[int] of var int: vars =
    [ apellido[n] | n in NOMBRE ]
 ++ [ musica[n]   | n in NOMBRE ]
 ++ [ edad[n]     | n in NOMBRE ];
 
%solve satisfy;

% solve :: int_search(vars, first_fail, indomain_min, complete) satisfy;

% solve :: int_search(vars, input_order, indomain_split, complete) satisfy;

% solve :: int_search(vars, input_order, indomain_min, complete) satisfy;

 solve :: int_search(vars, dom_w_deg, indomain_split, complete) satisfy;
% ----------------------- Salida ---------------------
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
