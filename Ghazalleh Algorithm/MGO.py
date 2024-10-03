import numpy as np

def MGO(N, MaxIter, LB, UB, dim, fobj):
    lb = np.ones(dim) * LB
    ub = np.ones(dim) * UB
    
    # Initialize the first random population of Gazelles
    X = initialization(N, dim, UB, LB)
    
    # Initialize Best Gazelle
    BestX = None
    BestFitness = float('inf')
    
    for i in range(N):
        # Calculate the fitness of the population
        Sol_Cost[i] = fobj(X[i])
        
        # Update the Best Gazelle if needed
        if Sol_Cost[i] <= BestFitness:
            BestFitness = Sol_Cost[i]
            BestX = X[i]
    
    # Main loop
    for Iter in range(MaxIter):
        for i in range(N):
            RandomSolution = np.random.permutation(N)[:int(np.ceil(N/3))]
            M = X[np.random.randint(int(np.ceil(N/3)), N)] * np.floor(np.random.rand()) + np.mean(X[RandomSolution], axis=0) * np.ceil(np.random.rand())
            
            # Calculate the vector of coefficients
            cofi = Coefficient_Vector(dim, Iter, MaxIter)
            A = np.random.randn(dim) * np.exp(2 - Iter * (2/MaxIter))
            D = (np.abs(X[i]) + np.abs(BestX)) * (2 * np.random.rand() - 1)
            
            # Update the location
            NewX = Solution_Imp(X, BestX, lb, ub, N, cofi, M, A, D, i)
            
            # Cost function calculation and Boundary check
            NewX, Sol_CostNew = Boundary_Check(NewX, fobj, LB, UB)
            
            # Adding new gazelles to the herd
            X = np.vstack((X, NewX))
            Sol_Cost = np.vstack((Sol_Cost, Sol_CostNew))
            idbest = np.argmin(Sol_Cost)
            BestX = X[idbest]