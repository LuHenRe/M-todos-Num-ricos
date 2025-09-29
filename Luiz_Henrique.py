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


"""
Retorna todos os valores diretamente com a equação B0 + B1 * x. Porém, utilizando coeficientes é possível fazer um por um
"""
def ajuste_curva(x, y):
    x = np.array(x)
    y = np.array(y)

    soma_x = np.sum(x)
    soma_y = np.sum(y)
    soma_x2 = np.sum(x ** 2)
    soma_xy = np.sum(x * y)
    n = len(x)

    m = np.array([[n, soma_x], [soma_x, soma_x2]])

    v = np.array([soma_y, soma_xy])

    coeficientes = np.linalg.solve(m, v)
    b0, b1 = coeficientes
    return b0 + b1 * x

"""
# Teste com arrays
x = np.array([0.2, 2.7, 4.5, 5.9, 7.2])
y = np.array([1.5, 1.8, 3.1, 2.6, 3.6])
"""




def qualidade_ajuste(x1, x2, y, x_1, x_2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    n = len(x1)

    soma_x1 = np.sum(x1)
    soma_x2 = np.sum(x2)
    soma_y = np.sum(y)
    soma_x1x2 = np.sum(x1 * x2)
    soma_x1y = np.sum(x1 * y)
    soma_x2y = np.sum(x2 * y)
    soma_x1_2 = np.sum(x1 ** 2)
    soma_x2_2 = np.sum(x2 ** 2)

    # Construção do sistema linear
    matriz = np.array([
        [n, soma_x1, soma_x2],
        [soma_x1, soma_x1_2, soma_x1x2],
        [soma_x2, soma_x1x2, soma_x2_2]
    ])

    vetor = np.array([soma_y, soma_x1y, soma_x2y])

    # Resolvendo o sistema linear para encontrar os coeficientes
    b0, b1, b2 = np.linalg.solve(matriz, vetor)

    # Função de predição
    def v(x1_val, x2_val):
        return b0 + b1 * x1_val + b2 * x2_val

    # Aplicar a função aos valores de entrada x_1 e x_2
    y_estimado = v(x_1, x_2)

    return y_estimado

"""
# Teste com arrays
x1 = np.array([-3, -2, 0, 1, 3, 5])
x2 = np.array([1, 2, 4, 5, 8, 9])
y = np.array([13, 18, 22, 27, 36, 39])

# Usar os próprios x1, x2 como entrada para prever os valores y
resultado = qualidade_ajuste(x1, x2, y, x1, x2)
print("Valores estimados de y:", resultado)
"""