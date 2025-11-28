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
'''
Só precisa do nome da função (f) quando declarar, sem o f(x)
'''


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
#Em fParada: no máximo, max_iter, iteracões ou diferença entre os limites do intervalo menor que tol.
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

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
            
    print(f"Número máximo de iterações ({max_iter}) alcançado.")
    return valores


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
resultado: array([1.39445328, 2.11595759, 2.63544069, 3.0394831 , 3.41466534])
"""

def u(x, b0, b1): return b0 + b1 * x
def graf_scatter(x, y, b0, b1):
    plt.scatter(x, y, color='blue')
    plt.plot(x, u(x, b0, b1), color='blue')
    plt.show()


def ajuste_curva_linear(x, y):
    # ... (todo o cálculo interno permanece o mesmo)
    m = np.array([[n, soma_x], [soma_x, soma_x2]])
    v = np.array([soma_y, soma_xy])

    coeficientes = np.linalg.solve(m, v)
    b0, b1 = coeficientes

    # Retorna os coeficientes em vez dos valores ajustados
    return b0, b1 

# Como usar a versão melhorada:
# x = np.array([0.2, 2.7, 4.5, 5.9, 7.2])
# y = np.array([1.5, 1.8, 3.1, 2.6, 3.6])
# b0, b1 = ajuste_curva_linear(x, y)
# print(f"Coeficientes: b0 = {b0}, b1 = {b1}")
# y_ajustado = b0 + b1 * x # Calcula os valores ajustados fora da função
# print(f"Valores ajustados: {y_ajustado}")


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


def regressao_polinomial(x, y): # Ajuste de curva
    x = np.array(x)  # x = np.array([0.5, 1.2, 2.1, 3.5, 5.4])
    y = np.array(y)  # y = np.array([5.1, 3.2, 2.8, 1, 0.4])
    m = np.array([[len(x), sum(x), sum(x ** 2), sum(x ** 3)],
                   [sum(x), sum(x ** 2), sum(x ** 3), sum(x ** 4)],
                   [sum(x ** 2), sum(x ** 3), sum(x ** 4), sum(x ** 5)],
                   [sum(x ** 3), sum(x ** 4), sum(x ** 5), sum(x ** 6)]])

    v = np.array([sum(y), sum(y * x), sum(y * x ** 2), sum(y * x ** 3)])

    coeficientes = np.linalg.solve(m, v)
    b0, b1, b2, b3 = coeficientes
    print(b0 + b1 * x + b2 * x ** 2, b3 * x ** 3)
    #[4.93674419 3.67886381 2.55717626 1.92051707 3.21434899] [-2.25088454e-03 -3.11162279e-02 -1.66763534e-01 -7.72053398e-01
    # -2.83546627e+00]
    u3 = b0 + b1 * x + b2 * (x ** 2) + b3 * (x ** 3)

    residuos = y - u3
    ss_res = np.sum(residuos ** 2)

    media_y = np.mean(y)
    ss_tot = np.sum((y - media_y) ** 2)

    r2_3grau = 1 - ss_res / ss_tot
    print(r2_3grau)
    # 0.9692408111274403



# Compara os ajustes em retas não retas.
# a e b são intervalos.
# x é o array/conjunto x utilizados nos cálculos de ajuste.
# u1, u2 e u3 são os u de cada tipo de ajuste
def comparacao_ajustes(a, b, x, u1, u2, u3):
    x1 = np.linspace(a, b, 50)  # x1 = np.linspace(0.5, 5.4, 50)
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
    # Gera 100 pontos para uma curva suave no intervalo dos dados
    x_curva = np.linspace(min(x_dados), max(x_dados), 100)

    # Calcula os valores de y para a curva suave de cada modelo
    # Ex: modelo1 é linear, modelo2 é quadrático, etc.
    b0_1, b1_1 = modelo1_coeffs
    y_curva1 = b0_1 + b1_1 * x_curva

    b0_2, b1_2, b2_2 = modelo2_coeffs
    y_curva2 = b0_2 + b1_2 * x_curva + b2_2 * (x_curva**2)

    b0_3, b1_3, b2_3, b3_3 = modelo3_coeffs
    y_curva3 = b0_3 + b1_3 * x_curva + b2_3 * (x_curva**2) + b3_3 * (x_curva**3)

    # Plota os dados originais como pontos
    plt.scatter(x_dados, y_dados, label='Dados Originais', color='red')

    # Plota as curvas de ajuste
    plt.plot(x_curva, y_curva1, label='Ajuste Linear')
    plt.plot(x_curva, y_curva2, label='Ajuste 2º Grau')
    plt.plot(x_curva, y_curva3, label='Ajuste 3º Grau')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparação dos Métodos de Ajuste')
    plt.grid(True)
    plt.legend()
    plt.show()



def IntegracaoSimples(f, a, b):
    return (f(a) + f(b)) / 2 * (b - a)

# a = limite inferior
# b = limite superior
def IntegracaoComposta(f, a, b, n):
    resultado = 0
    h = (b - a) / n
    max = b - a
    a, b, j = 0, 0, 0
    while j < max:
        b += h
        resultado += (f(a) + f(b)) / 2 * h
        j += h
        a += h
    return resultado


# Trapézio simples (n=1)
def trapezio_simples (fa, fb, h):
    I = h * (fa + fb) / 2 
    return I


def trapezio_composto(f, a, b, n):
    '''
    Calcula a integral f no intervalo [a, b] pela regra do trapézio.
    n = 1, trapézio simples
    '''
    def trapezio_s (fa, fb, h):
        I = h * (fa + fb) / 2 
        return I
    h = (b - a) / n
    x_barra = np.zeros(n + 1)

    i = 0
    while i < n + 1:
        x_barra[i] = a + i * h
        i += 1

    i = 0
    I = 0
    while i < n:
        I += trapezio_s(f(x_barra[i]), f(x_barra[i+1]), h)
        i += 1
    return I



import math

def normal_cdf_manual(z):
    """
    Calcula a FDC Normal Padrão N(z) usando uma aproximação polinomial.
    Baseado em Abramowitz e Stegun (1964), Fórmula 7.1.26.
    """
    
    # A aproximação funciona melhor para z >= 0, 
    # então usamos a simetria N(-z) = 1 - N(z)
    if z < 0:
        return 1 - normal_cdf_manual(-z)

    # Coeficientes da aproximação
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429

    # Termo auxiliar t = 1 / (1 + 0.2316419 * z)
    t = 1 / (1 + 0.2316419 * z)
    
    # O cálculo principal da aproximação Q(z) = 1 - N(z)
    Qz = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)

    # Retorna N(z) = 1 - Q(z)
    return 1 - Qz

# --- Testando a implementação manual ---
'''
print(f"N(0.0) manual: {normal_cdf_manual(0.0):.4f}")
print(f"N(1.0) manual: {normal_cdf_manual(1.0):.4f}")
print(f"N(-1.96) manual: {normal_cdf_manual(-1.96):.4f}")]
'''


def Interpolacao_Lagrange(x_pontos, y_pontos, x):
    n = len(x_pontos)
    total = 0.0
    
    for i in range(n):
        # L_i(x)
        Li = 1.0
        for j in range(n):
            if i != j:
                Li *= (x - x_pontos[j]) / (x_pontos[i] - x_pontos[j])
        total += y_pontos[i] * Li
    
    return total

# Exemplo de uso
'''
x = np.array([2, 2.2, 2.4, 2.5, 2.7, 2.9])
y = np.array([5.6569, 7.1789, 8.9234, 9.8821, 11.9787, 14.3217])

print(Interpolacao_Lagrange(x, y, 2.3))'''




def Diferenca_Newton(x, y):
    n = len(y)
    table = np.zeros((n, n))
    table[:,0] = y  # primeira coluna = y
    
    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = table[i+1][j-1] - table[i][j-1]
    
    return table

def Interpolacao_Newton(x_pontos, diff_table, x):
    n = len(x_pontos)
    result = diff_table[0][0]
    produto_termo = 1.0
    
    for i in range(1, n):
        produto_termo *= (x - x_pontos[i-1])
        result += diff_table[0][i] * produto_termo
    
    return result

# Exemplo de uso
'''
x = np.array([2, 2.2, 2.4, 2.5, 2.7, 2.9])
y = np.array([5.6569, 7.1789, 8.9234, 9.8821, 11.9787, 14.3217])

# criar tabela de diferenças
diff_table = Diferenca_Newton(x, y)

# interpolar ponto
print(Interpolacao_Newton(x, diff_table, 2.3))
'''


def Polinomio_Lagrange(xa, ya, x=np.linspace(0, 5, 100)):

    x0, x1, x2 = xa
    y0, y1, y2 = ya

    L2 = (y0 * ((x - x1)/(x0 - x1) * (x - x2)/(x0 - x2)) + 
          y1 * ((x - x0)/(x1 - x0) * (x - x2)/(x1 - x2)) + 
          y2 * ((x - x0)/(x2 - x0) * (x - x1)/(x2 - x1)))
    
    return L2

'''
xa = np.array([0.1, 0.6, 1.1])
ya = np.array([1.221, 3.320, 4.310])

x = np.linspace(0, 5, 100)
'''


import numpy as np

def PolinomioNewton(x, y):
    x0, x1, x2 = x
    y0, y1, y2 = y

    # Diferenças divididas
    fx01 = (y1 - y0) / (x1 - x0)
    fx12 = (y2 - y1) / (x2 - x1)
    fx012 = (fx12 - fx01) / (x2 - x0)

    # Construindo o polinômio expandido:
    # P(x) = y0 + fx01*(x - x0) + fx012*(x - x0)*(x - x1)
    # Vamos expandir isso para ax² + bx + c

    # Expansão do termo quadrático
    a = fx012

    # Expansão do termo linear
    b = fx01 - fx012*(x0 + x1)

    # Termo constante
    c = y0 - fx01*x0 + fx012*(x0*x1)

    return f"{a}x² + {b}x + {c}"


def diferencas_finitas(y):
    Dy = np.zeros(5)
    Dy2 = np.zeros(4)
    Dy3 = np.zeros(3)
    Dy4 = np.zeros(2)
    Dy5 = np.zeros(1)

    for i in range(5):
        Dy[i] = y[i+1] - y[i]

    for i in range(4):
        Dy2[i] = Dy[i+1] - Dy[i]

    for i in range(3):
        Dy3[i] = Dy2[i+1] - Dy2[i]

    for i in range(2):
        Dy4[i] = Dy3[i+1] - Dy3[i]
    Dy5[0] = Dy4[1] - Dy4[0]


def diferencas(y):
    tabela = [y.copy()]
    while len(tabela[-1]) > 1:
        coluna = tabela[-1]
        nova_coluna = np.diff(coluna)
        tabela.append(nova_coluna)
    return tabela

# Exemplo
'''
y = np.array([5.6569, 7.1789, 8.9234, 9.8821, 11.9787, 14.3217])

tabela = diferencas(y)

for i, col in enumerate(tabela):
    print(f"Δ^{i} y = {col}")
'''











#1) A)

   # Matematicamente, AMBOS geram EXATAMENTE O MESMO polinômio e o mesmo valor.
   # O Teorema da Unicidade garante que só existe um polinômio interpolador.
   # O gráfico e a estimativa serão idênticos nos dois métodos.

#2. DIFERENÇA PRÁTICA:
   
   #Lagrange
   #- Vantagem: Fórmula fechada e simétrica. Fácil de programar "bruto".
   #- Desvantagem: Se você precisar adicionar um novo ponto (n aumenta),
   #  precisa jogar tudo fora e recalcular do zero.

   #Newton
 # - Vantagem: Método incremental (construção em camadas).
 # Se adicionar um ponto, você aproveita o cálculo anterior.
 #  Gera a "Tabela de Diferenças Divididas".
  

#em questão de tempo , a melhor escolha seria o polinomio de newton , porque ele tem a 
# flexibilidade de adicionar um ponto sem ter que fazer o calculo do inico;
#e tambem ele faz a tabela de difirenças divididas com os coeficientes de b0 , b1 e etc...




#2 - C)
  # podemos dizer que nos resultado do 2A tinhamos varios pontos , automaticamente o tamanho da pista
  #  sera maior , quanto que no resultado  2B  foi analisado com apenas 3 pontos

  # automaticamente p em 84 a pista sera menor . O carro na pista com o comprimento maior(questão2A)
  #  devera ter a velocida maior para alcançar o fim da pista em 84 segundos, enquanto o comprimento 

  # com 3 pontos(2B) o carro vai ter uma velocidade menor comparada ao comprimento da questão 2A para
  #  percorrer a pista em 84 segundo.