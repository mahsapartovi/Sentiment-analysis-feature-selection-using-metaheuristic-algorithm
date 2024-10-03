import numpy as np

def initialization(N, dim, up, down):
    if up.shape[0] == 1:
        X = np.random.rand(N, dim) * (up - down) + down
    if up.shape[0] > 1:
        X = np.zeros((N, dim))
        for i in range(dim):
            high = up[i]
            low = down[i]
            X[:, i] = np.random.rand(1, N) * (high - low) + low
    return X