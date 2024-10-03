import numpy as np

def Coefficient_Vector(dim, Iter, MaxIter):
    a2 = -1 + Iter * ((-1) / MaxIter)
    u = np.random.randn(1, dim)
    v = np.random.randn(1, dim)
    cofi = np.zeros((4, dim))
    cofi[0, :] = np.random.rand(1, dim)
    cofi[1, :] = (a2 + 1) + np.random.rand()
    cofi[2, :] = a2 * np.random.randn(1, dim)
    cofi[3, :] = u * v**2 * np.cos((np.random.rand() * 2) * u)
    return cofi