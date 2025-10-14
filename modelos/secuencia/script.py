import subprocess

def ejecutar_secuencia_magica(n):
    # 1️⃣ Crear el archivo datos.dzn con el parámetro n
    with open("datos.dzn", "w") as archivo_datos:
        archivo_datos.write(f"n = {n};\n")

    print(f"📄 Archivo datos.dzn actualizado con n = {n}")

    # 2️⃣ Ejecutar MiniZinc con el modelo y los datos
    comando = ["minizinc", "--all-solutions", "secuencia.mzn", "datos.dzn"]

    resultado = subprocess.run(comando, capture_output=True, text=True)

    # 3️⃣ Mostrar salida o errores
    if resultado.returncode == 0:
        print("✅ Soluciones encontradas:\n")
        print(resultado.stdout)
    else:
        print("❌ Error al ejecutar MiniZinc:\n")
        print(resultado.stderr)

# 4️⃣ Permitir ejecutar desde la consola
if __name__ == "__main__":
    n = int(input("Introduce el valor de n: "))
    ejecutar_secuencia_magica(n)
