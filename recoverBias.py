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
    
    bias = 1/yTr[i_star] - K[i_star,:] @ (alphas * yTr)
        
    return bias 
    
def find_i_star(alphas,C):
    """
    Return index of element in alpha that is furthest from 0 and C
    """
    max_dist = abs(alphas[0] - C) + alphas[0]
    max_idx = 0
    for i in range(1,len(alphas)):
        if max_dist < abs(alphas[i] - C) + alphas[i]:
            max_dist = abs(alphas[i] - C) + alphas[i]
            max_idx = i
                   
    return max_idx