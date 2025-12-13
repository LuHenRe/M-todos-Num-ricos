import numpy as np

def Substituicao_Sucessiva_Inferior(l, b):
    '''
    Para matrizes triangulares inferiores
    '''
    n = len(b)
    x = np.zeros(n)
    if l.shape[0] != l.shape[1] or l.shape[0] != n:
        return 0
    for i in range(n):
        soma = np.dot(l[i, :i], x[:i])
        if l[i, i] == 0:
            return 0
        x[i] = (b[i] - soma) / l[i][i]
    return x



def Substituicao_Retroativa_Superior(l, b):
    '''
    Para matrizes triangulares superiores
    '''
    n = len(b)
    x = np.zeros(n)
    if l.shape[0] != l.shape[1] or l.shape[0] != n:
        return 0
    for i in range(n -1, -1, -1):
        # Nota: l aqui representa a matriz Upper (U)
        soma = np.dot(l[i, i+1:], x[i+1:])
        if l[i, i] == 0:
            return 0
        x[i] = (b[i] - soma) / l[i][i]
    return x



def eliminacao_gauss_pivoteamento(A, b):
    """
    Resolve o sistema linear Ax = b usando Eliminação de Gauss
    com Pivoteamento Parcial.
    """
    n = len(b)
    
    for k in range(n):
        max_index = k
        max_val = abs(A[k][k])
        
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                max_index = i
        
        # Se o maior valor for aproximadamente 0, a matriz é singular
        if max_val < 1e-10:
            return None # Não há solução única

        A[k], A[max_index] = A[max_index], A[k]
        b[k], b[max_index] = b[max_index], b[k]

        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            b[i] -= factor * b[k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]

    x = [0 for _ in range(n)]
    
    for i in range(n - 1, -1, -1):
        soma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - soma) / A[i][i]
        
    return x



def gauss_triangulacao_parcial(A_in, b_in):
    """
    Aplica Eliminação de Gauss com Pivoteamento Parcial
    RETORNA: Matriz Triangular Superior (U) e vetor modificado (b_mod)
    para serem usados na substituição retroativa.
    """
    # Copia para não alterar os originais fora da função
    A = np.copy(A_in).astype(float)
    b = np.copy(b_in).astype(float)
    n = len(b)
    
    for k in range(n-1):
        max_index = k + np.argmax(np.abs(A[k:, k]))
        
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]
            
        if abs(A[k, k]) < 1e-12:
            raise ValueError("Matriz singular ou quase singular detectada.")

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:] 
            b[i] -= factor * b[k]
            
            # Força zero exato na coluna k para evitar resíduos de float
            A[i, k] = 0.0

    return A, b