"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    
    i_star = find_i_star(alphas,C)
#     print((yTr[i_star]))
    bias =  K[i_star,:] @ (alphas * yTr) - yTr[i_star]
    print(bias[0])
    return bias 
    
def find_i_star(alphas,C):
    """
    Return index of element in alpha that is furthest from 0 and C
    0 < a < C
    """
    max_dist = -1
    max_idx = 0
        
    for i in range(len(alphas)):
        
        if alphas[i] >= C or alphas[i] <= 0:
            continue
        
        dist = abs(C/2.0 - alphas[i])
        
        if max_dist < dist:
            max_dist = dist
            max_idx = i
                   
    return max_idx