import numpy as np
import matplotlib.pyplot as plt

def F23(x):
    return np.sum(x**2)

def Get_Functions_details(Function_name):
    if Function_name == 'F23':
        lb = -100
        ub = 100
        dim = 30
        fobj = F23
    return lb, ub, dim, fobj

def MGO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj):
    Best_pos = None
    Best_score = np.inf
    MGO_cg_curve = np.zeros(Max_iteration)
    
    for i in range(Max_iteration):
        # Generate random solutions
        SearchAgents = np.random.uniform(lb, ub, (SearchAgents_no, dim))
        
        # Evaluate objective function
        scores = np.apply_along_axis(fobj, 1, SearchAgents)
        
        # Update best solution
        best_index = np.argmin(scores)
        if scores[best_index] < Best_score:
            Best_score = scores[best_index]
            Best_pos = SearchAgents[best_index]
        
        # Update convergence curve
        MGO_cg_curve[i] = Best_score
    
    return Best_score, Best_pos, MGO_cg_curve

SearchAgents_no = 30
Function_name = 'F23'
Max_iteration = 500

lb, ub, dim, fobj = Get_Functions_details(Function_name)
Best_score, Best_pos, MGO_cg_curve = MGO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)

plt.semilogy(MGO_cg_curve, color='r')
plt.title('Objective space')
plt.xlabel('Iteration')
plt.ylabel('Best score obtained so far')
plt.axis('tight')
plt.grid(True)
plt.box(True)
plt.legend(['MGO'])
print('The best solution obtained by MGO is:', Best_pos)
print('The best optimal value of the objective function found by MGO is:', Best_score)