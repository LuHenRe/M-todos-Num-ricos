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


def bissecao_iter(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        print("Erro: f(a) e f(b) devem ter sinais opostos.")
        return None

    valores = np.zeros(max_iter)
    for iteracao in range(max_iter):
        c = (a + b) / 2
        valores[iteracao] = c

        if f(c) == 0:
            print(f"Raiz exata encontrada em {c} na iteração {iteracao + 1}.")
            return valores[:iteracao + 1]
        
        if abs(b - a) < tol:
            print(f"Tolerância alcançada. Intervalo menor que {tol}.")
            return valores[:iteracao + 1]

        # Se f(a) e f(c) têm sinais opostos, a raiz está em [a, c].
        if f(a) * f(c) < 0:
            b = c
        # Caso contrário, a raiz está em [c, b].
        else:
            a = c
            
    print(f"Aviso: Número máximo de iterações ({max_iter}) alcançado.")
    return valores