import numpy as np
import matplotlib.pyplot as plt


def grafico(f, a, b):
    x = np.linspace(a, b, 100)
    plt.plot(x, f(x))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfico da função')
    plt.grid(True)
    plt.show()



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
    for i in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        a = b
        b = c
    return None



def falsa_posicao(f, a, b, tol=1e-6, max_iter=100):
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
    return None



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

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
            
    print(f"Número máximo de iterações ({max_iter}) alcançado.")
    return valores



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


def u(x, b0, b1): return b0 + b1 * x
def graf_scatter(x, y, b0, b1):
    plt.scatter(x, y, color='blue')
    plt.plot(x, u(x, b0, b1), color='blue')
    plt.show()


def ajuste_curva_linear(x, y):
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

    return b0, b1 



def ajuste_linear_multiplo(x1, x2, y, x_1, x_2):
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

    matriz = np.array([
        [n, soma_x1, soma_x2],
        [soma_x1, soma_x1_2, soma_x1x2],
        [soma_x2, soma_x1x2, soma_x2_2]
    ])

    vetor = np.array([soma_y, soma_x1y, soma_x2y])

    b0, b1, b2 = np.linalg.solve(matriz, vetor)

    def v(x1_val, x2_val):
        return b0 + b1 * x1_val + b2 * x2_val

    y_estimado = v(x_1, x_2)

    return y_estimado



def regressao_polinomial(x, y):
    x = np.array(x)
    y = np.array(y)
    m = np.array([[len(x), sum(x), sum(x ** 2), sum(x ** 3)],
                   [sum(x), sum(x ** 2), sum(x ** 3), sum(x ** 4)],
                   [sum(x ** 2), sum(x ** 3), sum(x ** 4), sum(x ** 5)],
                   [sum(x ** 3), sum(x ** 4), sum(x ** 5), sum(x ** 6)]])

    v = np.array([sum(y), sum(y * x), sum(y * x ** 2), sum(y * x ** 3)])

    coeficientes = np.linalg.solve(m, v)
    b0, b1, b2, b3 = coeficientes
    print(b0 + b1 * x + b2 * x ** 2, b3 * x ** 3)

    u3 = b0 + b1 * x + b2 * (x ** 2) + b3 * (x ** 3)

    residuos = y - u3
    ss_res = np.sum(residuos ** 2)

    media_y = np.mean(y)
    ss_tot = np.sum((y - media_y) ** 2)

    r2_3grau = 1 - ss_res / ss_tot
    print(r2_3grau)



def comparacao_ajustes(a, b, x, u1, u2, u3):
    x1 = np.linspace(a, b, 50)
    plt.plot(x, u1, label='Ajuste Mínimos Quadrados')
    plt.plot(x, u2, label='Ajuste 2º Grau')
    plt.plot(x, u3, label='Ajuste 3º Grau')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparação dos métodos de ajustes')
    plt.grid(True)
    plt.legend()
    plt.show()



def comparacao_ajustes_grafico(x_dados, y_dados, modelo1_coeffs, modelo2_coeffs, modelo3_coeffs):
    x_curva = np.linspace(min(x_dados), max(x_dados), 100)

    b0_1, b1_1 = modelo1_coeffs
    y_curva1 = b0_1 + b1_1 * x_curva

    b0_2, b1_2, b2_2 = modelo2_coeffs
    y_curva2 = b0_2 + b1_2 * x_curva + b2_2 * (x_curva**2)

    b0_3, b1_3, b2_3, b3_3 = modelo3_coeffs
    y_curva3 = b0_3 + b1_3 * x_curva + b2_3 * (x_curva**2) + b3_3 * (x_curva**3)

    plt.scatter(x_dados, y_dados, label='Dados Originais', color='red')

    plt.plot(x_curva, y_curva1, label='Ajuste Linear')
    plt.plot(x_curva, y_curva2, label='Ajuste 2º Grau')
    plt.plot(x_curva, y_curva3, label='Ajuste 3º Grau')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparação dos Métodos de Ajuste')
    plt.grid(True)
    plt.legend()
    plt.show()