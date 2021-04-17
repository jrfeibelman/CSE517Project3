import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the t distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
#     D = np.zeros((n, m))
    D = np.sqrt(np.sum(np.square(X - Z), axis=0))

#     ISSUE: np.sum compresses (100,100) to (100,1)
#     XZ = X - Z
#     SQ = XZ.T @ XZ
#     D = np.sum(SQ, axis=0)[:, np.newaxis]
#     D = np.sqrt(D)

#     ISSUE: np.sqrt(a+b+c=negative numbers)
#     a = np.sum(X**2, axis=0)[:, np.newaxis]
#     b = np.sum(Z**2, axis=0)[:, np.newaxis]
#     c = -2 * (a @ b.T)
#     D = np.sqrt(a+b+c)

    print(D.shape)
    return D
