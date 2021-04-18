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
    bias =  K[i_star,:] @ (alphas * yTr)
    print(bias[0])
    ratio = float(bias).as_integer_ratio()
    div = float(ratio[0] - yTr[i_star] * ratio[1]) / ratio[1]
    print(div)
    return div
    
def find_i_star(alphas,C):
    """
    Return index of element in alpha that is furthest from 0 and C
                or closest to C/2 
    0 < a < C
    """
    opt_dist = -1
    opt_idx = 0
    for i in range(len(alphas)):
        if alphas[i] >= C or alphas[i] <= 0:
            continue
        
        dist = abs(C/2.0 - alphas[i])
        
        if dist < opt_dist:
            opt_dist = dist
            opt_idx = i
                   
    return opt_idx