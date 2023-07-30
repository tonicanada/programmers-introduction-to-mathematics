import math
import matplotlib.pyplot as plt

def coeficiente_binomial(n, x):
    return math.factorial(n) // (math.factorial(x) * math.factorial(n - x))

def generar_grafica(n):
    valores_x = list(range(n + 1))
    valores_y = [coeficiente_binomial(n, x) for x in valores_x]

    plt.plot(valores_x, valores_y, marker='o')
    plt.xlabel('x')
    plt.ylabel('C(n, x)')
    plt.title(f'Coeficiente binomial C(n, x) para n = {n}')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    try:
        numero_n = int(input("Ingresa un número entero n: "))
        if numero_n < 0:
            print("El número debe ser mayor o igual a 0.")
        else:
            generar_grafica(numero_n)
    except ValueError:
        print("Error: Ingresa un número entero válido.")
