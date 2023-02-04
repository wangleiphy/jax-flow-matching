import numpy as np
from scipy.optimize import linear_sum_assignment

def periodic_distance(x0, x1, L):
    '''
    nearest image distance in the box
    '''
    n, dim = x0.shape
    rij = (np.reshape(x0, (n, 1, dim)) - np.reshape(x1, (1, n, dim)))
    rij = rij - L*np.rint(rij/L)
    return np.linalg.norm(rij, axis=-1) # (n, n)

def matching(x0, x1, L):
    '''
    sovles the assignment problem 
    '''
    cost_matrix = periodic_distance(x0, x1, L)    
    _, col_ind = linear_sum_assignment(cost_matrix)
    return x0, x1[col_ind, :]

if __name__=='__main__':
    n, dim = 32, 3  
    L = 1.234 

    np.random.seed(42)

    x0 = np.random.uniform(0, L, (n, dim))
    x1 = np.random.uniform(0, L, (n, dim))

    cost_matrix = periodic_distance(x0, x1, L)    
    print (np.trace(cost_matrix))

    x0, x1 = matching(x0, x1, L)

    cost_matrix = periodic_distance(x0, x1, L)    
    print (np.trace(cost_matrix))
