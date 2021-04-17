"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""
import numpy as np
from cvxopt import matrix

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]

    Q = (yTr.T * K) * yTr
    print(Q.shape)
    p = np.ones(n)
    G = np.identity(n)
    h = np.ones(n) * C
    A = yTr * np.identity(n)
    b = np.zeros(n)
            
    return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

