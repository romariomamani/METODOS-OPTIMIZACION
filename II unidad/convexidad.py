import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def es_convexa_una_variable(f, x):
    """ Verifica la convexidad de una función de una variable mediante la segunda derivada """
    segunda_derivada = sp.diff(f, x, 2)
    print(f"Segunda derivada: {segunda_derivada}")
    return sp.simplify(segunda_derivada) >= 0

def es_convexa_multivariable(funciones, variables):
    """ Verifica la convexidad de una función de múltiples variables usando la matriz Hessiana """
    hessiana = sp.hessian(funciones, variables)
    print("Matriz Hessiana:")
    sp.pprint(hessiana)
    return all(eigenvalue >= 0 for eigenvalue in hessiana.eigenvals().keys())

def graficar_funcion(f, x, intervalo=(-5, 5)):
    """ Grafica una función de una variable """
    f_lambd = sp.lambdify(x, f, 'numpy')
    x_vals = np.linspace(intervalo[0], intervalo[1], 400)
    y_vals = f_lambd(x_vals)
    plt.plot(x_vals, y_vals, label=str(f))
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Gráfico de la función")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x = sp.Symbol('x')
    f = x**2 + 3*x + 2
    convexa = es_convexa_una_variable(f, x)
    print("¿Es convexa?", convexa)
    graficar_funcion(f, x)
    x, y = sp.symbols('x y')
    g = x**2 + y**2 + x*y
    convexa_multi = es_convexa_multivariable(g, (x, y))
    print("¿Es convexa en múltiples variables?", convexa_multi)