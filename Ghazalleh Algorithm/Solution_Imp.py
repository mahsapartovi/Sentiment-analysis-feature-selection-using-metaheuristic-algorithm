import numpy as np

def Solution_Imp(X, BestX, lb, ub, N, cofi, M, A, D, i):
    NewX = np.zeros((4, len(X[0])))
    NewX[0, :] = (ub - lb) * np.random.rand() + lb
    NewX[1, :] = BestX - np.abs((np.random.randint(2) * M - np.random.randint(2) * X[i, :]) * A) * cofi[np.random.randint(4), :]
    NewX[2, :] = (M + cofi[np.random.randint(4), :]) + (np.random.randint(2) * BestX - np.random.randint(2) * X[np.random.randint(N), :]) * cofi[np.random.randint(4), :]
    NewX[3, :] = (X[i, :] - D) + (np.random.randint(2) * BestX - np.random.randint(2) * M) * cofi[np.random.randint(4), :]
    return NewX