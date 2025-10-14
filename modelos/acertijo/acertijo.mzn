% --------------------------------------------
% acertijo.mzn — Acertijo lógico musical
% --------------------------------------------

set of int: PERSONAS = 1..3;
enum NOMBRE = {Juan, Oscar, Dario};
enum APELLIDO = {Gonzalez, Garcia, Lopez};
enum MUSICA = {Clasica, Pop, Jazz};

array[PERSONAS] of var NOMBRE: nombre;
array[PERSONAS] of var APELLIDO: apellido;
array[PERSONAS] of var MUSICA: musica;
array[PERSONAS] of var 24..26: edad;

% Todos diferentes
constraint alldifferent(apellido);
constraint alldifferent(musica);
constraint alldifferent(edad);

% 1. Juan es más viejo que González, a quien le gusta la música clásica.
constraint forall(i,j in PERSONAS where nombre[i] = Juan /\ apellido[j] = Gonzalez)(
    edad[i] > edad[j] /\ musica[j] = Clasica
);

% 2. El fan de la música pop no es García ni tiene 24 años.
constraint forall(i in PERSONAS where musica[i] = Pop)(
    apellido[i] != Garcia /\ edad[i] != 24
);

% 3. Oscar, quien no es López, tiene 25 años.
constraint forall(i in PERSONAS where nombre[i] = Oscar)(
    apellido[i] != Lopez /\ edad[i] = 25
);

% 4. La música favorita de Darío no es el jazz.
constraint forall(i in PERSONAS where nombre[i] = Dario)(
    musica[i] != Jazz
);

solve satisfy;

output [
  "Persona ", show(i), ": ",
  show(nombre[i]), " ", show(apellido[i]), ", ",
  show(edad[i]), " años, gusta de ", show(musica[i]), "\n",
  | i in PERSONAS
];