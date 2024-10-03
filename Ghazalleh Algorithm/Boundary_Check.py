def boundaryCheck(x, lb, ub):
    for i in range(len(x)):
        if x[i] < lb[i]:
            x[i] = lb[i]
        elif x[i] > ub[i]:
            x[i] = ub[i]
    return x

def boundary_Check(NewX, fobj, LB, UB):
    Sol_CostNew = []
    for j in range(4):
        NewX[j] = boundaryCheck(NewX[j], LB, UB)
        Sol_CostNew.append(fobj(NewX[j]))
    return NewX, Sol_CostNew