import numpy as np
import matplotlib.pyplot as plt


# =============================
#       FUNÇÕES DE GRÁFICO
# =============================

def Grafico(x, y, title='Gráfico', nmx='x', nmy='y'):
    """
    Exibe um gráfico simples de y em função de x.
    """
    plt.plot(x, y)
    plt.title(title)
    plt.grid(True)
    plt.xlabel(nmx)
    plt.ylabel(nmy)
    plt.show()


def Grafal(x0, x1, func, title='Gráfico', nmx='x', nmy='y'):
    """
    Plota uma função matemática f(x) no intervalo [x0, x1].
    """
    x = np.linspace(x0, x1, 500)
    y = func(x)
    plt.plot(x, y)
    plt.title(title)
    plt.grid(True)
    plt.xlabel(nmx)
    plt.ylabel(nmy)
    plt.show()


def Graficos(func, title='Gráfico', nmy='y'):
    """
    Gera vários gráficos da mesma função com diferentes níveis de zoom.
    """
    doms = [
        np.linspace(-1000, 1000, 500),
        np.linspace(-100, 100, 500),
        np.linspace(-50, 50, 500),
        np.linspace(-20, 20, 500),
        np.linspace(-10, 10, 500),
        np.linspace(-5, 5, 500),
        np.linspace(-2, 2, 500),
        np.linspace(-1, 1, 500),
        np.linspace(-0.5, 0.5, 500)
    ]
    fig, axis = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))
    plt.suptitle(title)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.90, hspace=0.5)
    axis = axis.flatten()
    for i, x in enumerate(doms):
        axis[i].plot(x, func(x))
        axis[i].grid()
        axis[i].set_title(f"Zoom {x[0]} a {x[-1]}")
        axis[i].set_ylabel(nmy)
    plt.show()


# =============================
#    MÉTODOS NUMÉRICOS
# =============================

def Bissecao(f, a, b, max_iteracoes=100, tolerancia=1e-6, num=1):
    """
    Encontra uma raiz de f(x) no intervalo [a, b] usando o método da bisseção.
    """
    iteracao = 0
    if f(a) * f(b) >= 0:
        print("A função deve ter sinais opostos nos extremos do intervalo.")
    else:
        while (b - a) / 2 > tolerancia and iteracao < max_iteracoes:
            c = (a + b) / 2
            if f(c) == 0:
                break
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
            iteracao += 1
        print(f"Raiz {num} aproximada: {(a + b) / 2}")
        print(f"Iterações: {iteracao}")


def Secante(f, x0, x1, tol=1e-4, max_inter=100):
    """
    Calcula a raiz de f(x) pelo método da secante.
    """
    for i in range(max_inter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2
        x0 = x1
        x1 = x2
    return None


def falsa_posicao(f, a, b, tol=1e-6, max_iter=100):
    """
    Encontra a raiz de f(x) no intervalo [a, b] pelo método da falsa posição.
    """
    if f(a) * f(b) >= 0:
        print("f(a) e f(b) devem ter sinais opostos")
        return None
    for _ in range(max_iter):
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return None


# =============================
#       REGRESSÃO LINEAR
# =============================

def Regressao_linear_matriz(x, y):
    """
    Realiza regressão linear simples por meio da resolução matricial.
    Retorna os coeficientes b0 e b1 da reta ajustada.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    soma_x = np.sum(x)
    soma_x2 = np.sum(x**2)
    soma_y = np.sum(y)
    soma_xy = np.sum(x * y)
    A = np.array([[n, soma_x], [soma_x, soma_x2]])
    b = np.array([soma_y, soma_xy])
    b0, b1 = np.linalg.solve(A, b)
    return b0, b1


def r(x, b0, b1):
    """
    Retorna o valor estimado de y dado x, b0 e b1.
    """
    return b0 + b1 * x


def Rquadrado(y_real, y_estimado):
    """
    Calcula o coeficiente de determinação R² entre valores reais e estimados.
    """
    SSres = np.sum((y_real - y_estimado) ** 2)
    media_y = np.mean(y_real)
    SStot = np.sum((y_real - media_y) ** 2)
    return 1 - SSres / SStot


# =============================
#   MÍNIMOS QUADRADOS GERAIS
# =============================

def minimosquadrados(x, y, grau=1):
    """
    Ajusta um polinômio de grau especificado aos dados (x, y) usando mínimos quadrados.

    Exibe a equação ajustada e o gráfico do ajuste.
    Retorna os coeficientes do polinômio.
    """
    coef = np.polyfit(x, y, grau)
    p = np.poly1d(coef)

    if grau == 1:
        print(f'Equação ajustada: y = {coef[0]:.4f}x + {coef[1]:.4f}')
    else:
        print(f'Equação ajustada (grau {grau}):')
        print(p)

    plt.scatter(x, y, color='blue', label='Dados')
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, p(x_vals), color='red', label=f'Ajuste grau {grau}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Ajuste de Mínimos Quadrados')
    plt.grid(True)
    plt.show()

    return coef


def avaliar_ajuste_minimos(coef, x_val):
    """
    Avalia um polinômio ajustado (via mínimos quadrados) para novos valores de x.
    """
    p = np.poly1d(coef)
    return p(x_val)


def coeficiente_determinacao_equacao(x, y, grafico=False):
    """
    Calcula o coeficiente de determinação R² para uma regressão quadrática.
    Retorna R² e a equação do ajuste, ou exibe o gráfico se grafico=True.
    """
    matriz = np.array([
        [len(x), np.sum(x), np.sum(x**2)],
        [np.sum(x), np.sum(x**2), np.sum(x**3)],
        [np.sum(x**2), np.sum(x**3), np.sum(x**4)]
    ])
    vetor = np.array([[np.sum(y)], [np.sum(x*y)], [np.sum(x**2*y)]])
    coeficientes = np.linalg.solve(matriz, vetor)
    b0, b1, b2 = coeficientes.flatten()
    u = b0 + b1*x + b2*x**2
    R = np.sum((y-u)**2)
    r_quadrado = 1 - (R /(np.sum(y**2)-(np.sum(y)**2/len(x))))

    if grafico is False:
        equacao_reta = f"{b0:.2f} + ({b1:.2f})x + ({b2:.2f})x²"
        return (r_quadrado, equacao_reta)
    else:
        plt.scatter(x, y, color='blue', label='Pontos')
        plt.plot(x, b0 + b1*x + b2*x**2, color="red",
                 label=f"y={b0:.2f} + {b1:.2f}x + {b2:.2f}x²")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Polinômio de Segundo Grau")
        plt.grid(True)
        plt.show()


def coef_det(x, y):
    """
    Calcula o coeficiente de determinação R² para uma regressão quadrática.
    """
    matriz = np.array([
        [len(x), np.sum(x), np.sum(x**2)],
        [np.sum(x), np.sum(x**2), np.sum(x**3)],
        [np.sum(x**2), np.sum(x**3), np.sum(x**4)]
    ])
    vetor = np.array([[np.sum(y)], [np.sum(x*y)], [np.sum(x**2*y)]])
    coeficientes = np.linalg.solve(matriz, vetor)
    b0, b1, b2 = coeficientes.flatten()
    u = b0 + b1*x + b2*x**2
    R = np.sum((y-u)**2)
    return 1 - (R /(np.sum(y**2)-(np.sum(y)**2/len(x))))


def regressao_linear_multipla(x1, x2, y):
    """
    Realiza regressão linear múltipla para duas variáveis independentes x1 e x2.
    Retorna coeficientes, R² e a equação ajustada.
    """
    X = np.column_stack((np.ones(len(x1)), x1, x2))
    XtX = X.T @ X
    Xty = X.T @ y
    coef = np.linalg.solve(XtX, Xty)
    b0, b1, b2 = coef

    y_pred = b0 + b1 * x1 + b2 * x2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    equacao = f"y = {b0:.5f} + ({b1:.5f})x_1 + ({b2:.5f})x_2"
    return coef, r2, equacao



##FUNCOES PARA A SEGUNDA PROVA

def tabela_diferencas_divididas(x_pontos, y_pontos):
    """
    Calcula e IMPRIME a tabela completa de Diferenças Divididas.
    Retorna a lista de coeficientes (b0, b1, b2...) para uso posterior.
    """
    n = len(x_pontos)
    if n != len(y_pontos):
        print("ERRO: Listas de tamanhos diferentes.")
        return None

    # 1. Cria a matriz n x n preenchida com zeros
    tabela = [[0] * n for _ in range(n)]
    
    # 2. Preenche a primeira coluna com os valores de Y
    for i in range(n):
        tabela[i][0] = y_pontos[i]

    # 3. Calcula as diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            numerador = tabela[i+1][j-1] - tabela[i][j-1]
            denominador = x_pontos[i+j] - x_pontos[i]
            
            if denominador == 0:
                print("Erro: Divisão por zero (pontos X repetidos).")
                return None
                
            tabela[i][j] = numerador / denominador

    # 4. Imprime a Tabela Formatada (Para copiar na prova)
    print(f"\n=== TABELA DE DIFERENÇAS DIVIDIDAS ({n} Pontos) ===")
    print(f"{'X':<8} | {'Y (Ordem 0)':<12} | {'Ordem 1':<12} | {'Ordem 2':<12} | ...")
    print("-" * 60)

    for i in range(n):
        linha_str = f"{x_pontos[i]:<8.4f} | "
        for j in range(n - i):
            linha_str += f"{tabela[i][j]:<12.4f} | "
        print(linha_str)
    print("-" * 60)

    # 5. Pega a diagonal principal (os coeficientes b0, b1, b2...)
    coeficientes = [tabela[0][j] for j in range(n)]
    
    print(f"Coeficientes (b0, b1...): {coeficientes}")
    return coeficientes


#############################################################################################################################

def interpolacao_newton(x_pontos, y_pontos, x_eval=None):
    """
    Realiza a Interpolação Polinomial de Newton (Diferenças Divididas).
    - Calcula os coeficientes automaticamente.
    - Plota o gráfico.
    - Se x_eval for passado: Retorna o VALOR numérico.
    - Se x_eval for None: Retorna a FUNÇÃO P(x).
    """
    n = len(x_pontos)
    if n != len(y_pontos):
        print("ERRO: As listas X e Y devem ter o mesmo tamanho.")
        return None

    # 1. Cálculo dos Coeficientes (Tabela de Diferenças Divididas)
    tabela = [[0] * n for _ in range(n)]
    for i in range(n):
        tabela[i][0] = y_pontos[i]

    for j in range(1, n):
        for i in range(n - j):
            numerador = tabela[i+1][j-1] - tabela[i][j-1]
            denominador = x_pontos[i+j] - x_pontos[i]
            if denominador == 0: return 0
            tabela[i][j] = numerador / denominador

    # A primeira linha da tabela contém os coeficientes b0, b1, b2...
    coeficientes = [tabela[0][j] for j in range(n)]

    # 2. Definição da Função Polinomial P(x)
    def P(x_valor):
        # Começa pelo último termo e vem voltando (Algoritmo de Horner otimizado)
        resultado = coeficientes[n-1]
        for i in range(n-2, -1, -1):
            resultado = resultado * (x_valor - x_pontos[i]) + coeficientes[i]
        return resultado

    print(f"=== INTERPOLAÇÃO DE NEWTON ({n} PONTOS) ===")
    print(f"Coeficientes calculados: {['{:.4f}'.format(c) for c in coeficientes]}")

    # 3. Impressão do Resultado
    val_estimado = None
    if x_eval is not None:
        val_estimado = P(x_eval)
        print(f"> P({x_eval}) = {val_estimado:.6f}")
    else:
        print("Polinômio gerado com sucesso.")

    try:
        x_plot = np.linspace(min(x_pontos), max(x_pontos), 500)
        y_plot = [P(x) for x in x_plot]

        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, y_plot, label='Polinômio Newton P(x)', color='orange', linewidth=2)
        plt.scatter(x_pontos, y_pontos, color='red', s=100, zorder=5, label='Dados')
        
        if x_eval is not None:
            if min(x_pontos) <= x_eval <= max(x_pontos):
                 plt.scatter([x_eval], [val_estimado], color='black', s=150, marker='X', zorder=6, label=f'Ponto {x_eval}')

        plt.title(f"Grau {n-1} (Diferenças Divididas)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Erro no gráfico: {e}")

    # RETORNO INTELIGENTE
    if x_eval is not None:
        return val_estimado  # Devolve o NÚMERO
    else:
        return P             # Devolve a FUNÇÃO


###################################################################################################################################


def analise_interpolacao_lagrange(x_pontos, y_pontos, x_eval=None):
    """
    Versão Mista:
    - Se pediu ponto (x_eval): Retorna o NÚMERO (resultado).
    - Se NÃO pediu ponto: Retorna a FUNÇÃO (P).
    """
    n = len(x_pontos)
    if n != len(y_pontos):
        print("ERRO: As listas X e Y devem ter o mesmo tamanho.")
        return None

    # 1. Definição Matemática
    def P(x_valor):
        soma_total = 0.0
        for i in range(n):
            y_i = y_pontos[i]
            produto_Li = 1.0
            for j in range(n):
                if i != j:
                    numerador = x_valor - x_pontos[j]
                    denominador = x_pontos[i] - x_pontos[j]
                    if denominador == 0: return 0 
                    produto_Li = produto_Li * (numerador / denominador)
            soma_total = soma_total + (y_i * produto_Li)
        return soma_total

    print(f"=== INTERPOLAÇÃO DE LAGRANGE ({n} PONTOS) ===")
    
    # 2. Gráfico e Resultado
    resultado = None
    if x_eval is not None:
        resultado = P(x_eval)
        print(f"> P({x_eval}) = {resultado:.6f}")

    try:
        x_plot = np.linspace(min(x_pontos), max(x_pontos), 500)
        y_plot = [P(x) for x in x_plot]

        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, y_plot, label='Polinômio P(x)', color='blue')
        plt.scatter(x_pontos, y_pontos, color='red', s=100, zorder=5, label='Dados')
        
        if x_eval is not None:
             if min(x_pontos) <= x_eval <= max(x_pontos):
                 plt.scatter([x_eval], [resultado], color='lime', s=150, marker='X', zorder=6, label='Estimativa')

        plt.title(f"Grau {n-1}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Erro no gráfico: {e}")

    # --- O SEGREDO ESTÁ AQUI ---
    if x_eval is not None:
        return resultado  # Devolve o NÚMERO (para o seu print funcionar)
    else:
        return P          # Devolve a FUNÇÃO (para usar depois)


#####################################################################################################################

def regra_trapezio(funcao, a, b, n=4, tolerancia=0.00001):
    """
    Resolve a questão completa de Integração Numérica.
    
    Parâmetros:
      funcao: A equação (def ou lambda)
      a: Início do intervalo
      b: Fim do intervalo
      n: O número de divisões 
      tolerancia: Precisão para a busca automática
    """
    
    # --- Lógica de Cálculo ---
    def calcular_trapezio(n_divisoes):
        h = (b - a) / n_divisoes
        soma = funcao(a) + funcao(b)
        for i in range(1, n_divisoes):
            soma += 2 * funcao(a + i * h)
        return (h / 2) * soma

    print(f"(Intervalo [{a}, {b}]) ")

  
    res_simples = calcular_trapezio(1)
    print(f"\n1. Trapézio Simples:")
    print(f"   RESULTADO: {res_simples:.6f}")
    
    #  Trapézio Composto (Com o n definido no parâmetro)
    res_composto = calcular_trapezio(n)
    print(f"\n2. Trapézio Composto (n={n}):")
    print(f"   RESULTADO: {res_composto:.6f}")


    try:
        x_vals = np.linspace(a, b, 200)
        y_vals = funcao(x_vals)
        plt.figure(figsize=(8, 3))
        plt.plot(x_vals, y_vals, label='Função')
        plt.fill_between(x_vals, y_vals, alpha=0.3, color='cyan', label='Área')
        plt.title("Gráfico da Função")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        print("(Não foi possível gerar o gráfico, prosseguindo...)")

    # --- PARTE 3: Valor Exato (Convergência) ---

    n_busca = 1
    I_anterior = calcular_trapezio(n_busca)
    
    while True:
        n_busca *= 2
        I_atual = calcular_trapezio(n_busca)
        
        if I_atual == 0: erro = 0
        else: erro = abs(I_atual - I_anterior) / abs(I_atual)
            
        if erro < tolerancia: break
        I_anterior = I_atual
        
    print(f"   RESULTADO DA INTEGRAL: {I_atual:.8f}")
    print(f"   MELHOR N ENCONTRADO= {n_busca}")
    
###################################################################################################

def regra_simpson(funcao, a, b, n=4, tolerancia=0.00001):
    """
    Calcula a integral usando a 1ª Regra de Simpson (1/3).
    IMPORTANTE: O número de intervalos 'n' DEVE ser PAR.
    """
    
    # Função interna de cálculo
    def calcular_simpson(n_divs):
        # Validação de N par
        if n_divs % 2 != 0:
            return None # Erro matemático
            
        h = (b - a) / n_divs
        
        # Extremos
        soma = funcao(a) + funcao(b)
        
        # Ímpares (multiplica por 4)
        soma_impares = 0
        for i in range(1, n_divs, 2):
            soma_impares += funcao(a + i * h)
            
        # Pares (multiplica por 2)
        soma_pares = 0
        for i in range(2, n_divs, 2):
            soma_pares += funcao(a + i * h)
            
        return (h / 3) * (soma + 4*soma_impares + 2*soma_pares)

    print(f"=== REGRA DE SIMPSON 1/3 (Intervalo [{a}, {b}]) ===")

    # 1. Cálculo com o n pedido
    if n % 2 != 0:
        print(f"AVISO: Simpson exige N par. Seu n={n} foi ajustado para n={n+1}.")
        n += 1
    
    resultado_n = calcular_simpson(n)
    print(f"\n1. Resultado para n={n}:")
    print(f"   Integral ≈ {resultado_n:.6f}")

    # 2. Gráfico
    try:
        x_vals = np.linspace(a, b, 200)
        y_vals = funcao(x_vals)
        plt.figure(figsize=(8, 3))
        plt.plot(x_vals, y_vals, label='Função', color='magenta')
        plt.fill_between(x_vals, y_vals, alpha=0.2, color='magenta', label='Área Simpson')
        plt.title(f"Integração Numérica")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        pass

    # 3. Convergência (Melhor precisão)
    print("\n2. Buscando melhor precisão...")
    n_busca = 2 # Começa com o menor par possível
    I_ant = calcular_simpson(n_busca)
    
    while True:
        n_busca *= 2
        I_atual = calcular_simpson(n_busca)
        
        if I_atual == 0: erro = 0
        else: erro = abs(I_atual - I_ant) / abs(I_atual)
            
        if erro < tolerancia: break
        I_ant = I_atual
        
    print(f"   Melhor Resultado: {I_atual:.8f}")
    print(f"   (Convergência com n={n_busca})")
    print("="*45)
    
    return resultado_n

######################################################################################



def regra_simpson_38(funcao, a, b, n=3, tolerancia=0.00001):
    """
    Calcula a integral usando a Regra de Simpson 3/8.
    """
    
    
    def calcular_38(n_divs):
        if n_divs % 3 != 0: return None 
        h = (b - a) / n_divs
        soma = funcao(a) + funcao(b)
        for i in range(1, n_divs):
            x_atual = a + i * h
            if i % 3 == 0:
                soma += 2 * funcao(x_atual)
            else:
                soma += 3 * funcao(x_atual)   
        return (3 * h / 8) * soma

    print(f"=== REGRA DE SIMPSON 3/8 (Intervalo [{a}, {b}]) ===")

    # Validação do N
    if n % 3 != 0:
        novo_n = n + (3 - (n % 3))
        print(f"AVISO: Simpson 3/8 exige N múltiplo de 3.")
        print(f"       Seu n={n} foi ajustado para n={novo_n}.")
        n = novo_n
    
    tipo = "SIMPLES" if n == 3 else "COMPOSTA"
    resultado_n = calcular_38(n)
    
    print(f"\n1. Resultado para n={n} ({tipo}):")
    print(f"   Integral ≈ {resultado_n:.6f}")

    # Gráfico
    try:
        x_vals = np.linspace(a, b, 200)
        y_vals = funcao(x_vals)
        plt.figure(figsize=(8, 3))
        plt.plot(x_vals, y_vals, label='Função', color='green')
        plt.fill_between(x_vals, y_vals, alpha=0.2, color='lime', label='Área 3/8')
        plt.title(f"Integração Simpson 3/8 (n={n})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except:
        pass

    # Convergência
    print("\n2. Buscando precisão máxima...")
    n_busca = 3 
    I_ant = calcular_38(n_busca)
    
    while True:
        n_busca *= 2 
        I_atual = calcular_38(n_busca)
        if I_atual == 0: erro = 0
        else: erro = abs(I_atual - I_ant) / abs(I_atual)
        if erro < tolerancia: break
        I_ant = I_atual
        
    print(f"   Melhor Resultado: {I_atual:.8f}")
    print(f"   (Atingido com n={n_busca})")
    print("="*45)

    return resultado_n