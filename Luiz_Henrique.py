import numpy as np
import matplotlib.pyplot as plt

'''
Faz um gráfico para uma função específica com a variável 'f' e 'a, b' como pontos do eixo das abscissas do gráfico 
'''
def grafico(f, a, b):
    x = np.linspace(a, b, 100)
    plt.plot(x, f(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfico da função')
    plt.grid(True)
    plt.show()


"""
Faz a bisseção de um intervalo [a, b] para encontrar uma raiz real
"""
def bissecao(funcao, a, b, tol=1e-4):
    raiz = (a + b) / 2

    while abs(funcao(raiz)) > tol:
        if funcao(a) * funcao(raiz) < 0:
            b = raiz
        else:
            a = raiz

        raiz = (a + b) / 2

    print(f"A raiz da equação é: {raiz}\nFunção da raiz: {funcao(raiz)}")


def secante(f, a, b, tol=1e-6, max_iter=100):
#Encontra a raiz, no intervalo [x0, x1], da equação definida 
#em fParada: no máximo, max_iter, iteracões ou diferença entre
#os limites do intervalo menor que tol.
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        a = b
        b = c
    return None # Não converge


def falsa_posicao(f, a, b, tol=1e-6, max_iter=100):
#Encontra a raiz, no intervalo [x0, x1], da equação definida em fParada: no máximo, max_iter, iterações ou diferença entre
#os limites do intervalo menor que tol.
    if f(a) * f(b) >= 0:
        print("ops! f(a) e f(b) devem ter sinais opostos")
        return None
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return None # não converge