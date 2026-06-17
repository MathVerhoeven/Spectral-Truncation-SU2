import sympy as sp
from sympy.physics.wigner import clebsch_gordan as sp_cg
import numpy as np


#Compute the dimension the propagated operator system R_J^{\circ 2}, with J = NN/2

#Tested for 0 <= NN <= 4
NN = 2

#########################################

def qq(x,y):
    '''Converts the pair (x,y) to a sympy rational representing x/y'''
    return sp.Rational(int(x),y)

def CG(j,j0,j1,m,m0,m1):
    '''Computes the Clebsch-Gordan coefficient C^{j,j0,j1}_{m,m0,m1} using the implementation in sympy'''
    
    # Convert python floats to sympy rationals
    j = qq(2*j,2)
    j0 = qq(2*j0,2)
    j1 = qq(2*j1,2)
    m = qq(2*m,2)
    m0 = qq(2*m0,2)
    m1 = qq(2*m1,2)    
    
    
    val = sp_cg(j, j0, j1, m, m0, m1)
    return val

def spectralprojCGnaive(n,j,m,mp):
    '''Computes the Peter-Weyl projection Q_J f^{j}_{m,mp} Q_J with J = n/2.
    
    Index order of increasing: mp -> m -> n, so e.g. 
    [(0,0,0), (1/2,-1/2,-1/2), (1/2,-1/2,1/2),
    (1/2,1/2,-1/2),(1/2,1/2,1/2)]
    '''
    # Naive implementation: No effort is made to use the special structure of CG coefficients for speedup or memory saving.

    indexlist = [(n1/2,k1-n1/2,k1p-n1/2) for n1 in range(0,n+1) for k1 in range(0,n1+1) for k1p in range(0,n1+1)]

    bigmat = [[(CG(j,t[0],r[0],m,t[1],r[1])*CG(j,t[0],r[0],mp,t[2],r[2])*np.sqrt((2*j+1)*(2*t[0]+1)/(2*r[0]+1))) for t in indexlist] for r in indexlist]
    return np.array(bigmat,dtype=float)

def all_matrix_products(matrices):
    '''Converts a list of 2d numpy arrays (matrices) to a list of 1d numpy arrays (vectors) containing the pairwise products of matrices flattened to a 1d array.'''

    n = len(matrices)
    products = []

    for i in range(n):
        for j in range(n):
            products.append((np.array(matrices[i]) @ np.array(matrices[j])).flatten())
    
    return products # a list of vectors

def span_dimension(vectors):
    '''Computes the dimension of the span for a list of 1d arrays (vectors). This is equal to the rank of the matrix with these vectors as rows.'''

    matrix = np.array(vectors, dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    return rank


### Initialize basis matrices of R_J, J = NN/2
expecteddim = ((NN+1)*(NN+2)*(2*NN+3)//6)**2

print('Computing basis matrices of R_J')
matsRJ = []
for n in range(0,2*NN+1):
    print(f'j = {n/2} out of {NN}')
    for k in range(0,n+1):
        for l in range(0,n+1):
            matsRJ += [spectralprojCGnaive(NN,n/2,k-n/2,l-n/2)]
print('Basis matrices computed')

### Compute products of these matrices

RJprop2 = all_matrix_products(matsRJ)
print('\n')
print('dim of R_J propagated = '+ str(span_dimension(RJprop2)))
print('dim of C*env = '+str(expecteddim))
