def boundary_check(X, lb, ub):
    for i in range(X.shape[0]):
        FU = X[i, :] > ub
        FL = X[i, :] < lb
        X[i, :] = (X[i, :] * ~(FU + FL)) + ub * FU + lb * FL
    return X