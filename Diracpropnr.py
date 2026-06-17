import sympy as sp
from sympy.physics.wigner import clebsch_gordan as sp_cg
import numpy as np
from numpy import sqrt as sqrt #used to be from math
from random import shuffle as shuffle

#Compute the dimension of propagated operator system of S_J, with J = NN/2

#Tested for 0 <= NN <= 2
NN = 1



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

def spinorinp(j,m,mp,p1,j1,m1,m1p,p2,j2,m2,m2p):
    '''Implementation of the inner product of matrix coefficient with spinors. Specifically, the inner product <f^{j}_{m,mp} psi^{p1 j1}_{m1,m1p}, psi^{p2 j2}_{m2,m2p}>.
    Here j,m,mp are half-integers, while p is either +1 or -1 and denotes positive or negative eigenspinors.
    '''
    if (p1 > 0) and (p2 > 0):
        pref = sqrt(2*j+1)*CG(j,j1,j2,m,m1,m2)/(2*j2+1)
        rest = sqrt(j1-m1p)*sqrt(j2-m2p)* CG(j,j1,j2,mp,m1p+1,m2p+1) + sqrt(j1+1+m1p)*sqrt(j2+1+m2p)*CG(j,j1,j2,mp,m1p,m2p)
    
    if (p1 < 0) and (p2 > 0):
        pref = sqrt(2*j+1)*CG(j,j1+1/2,j2,m,m1-1/2,m2)/(2*j2+1)
        rest = -sqrt(j1 + 1 + m1p)*sqrt(j2 - m2p)*CG(j,j1+1/2,j2,mp,m1p+1/2,m2p+1) + sqrt(j1+1-m1p)*sqrt(j2+1+m2p)*CG(j,j1+1/2,j2,mp,m1p-1/2,m2p)

    if (p1 > 0) and (p2 < 0):
        pref = sqrt(2*j+1)*CG(j,j1,j2+1/2,m,m1,m2-1/2)/(2*j2+2)
        rest = -sqrt(j1 - m1p)*sqrt(j2 + 1 + m2p)*CG(j,j1,j2+1/2,mp,m1p+1,m2p+1/2) + sqrt(j1+1+m1p)*sqrt(j2+1-m2p)*CG(j,j1,j2+1/2,mp,m1p,m2p-1/2)
    
    if (p1 < 0) and (p2 < 0):
        pref = sqrt(2*j+1)*CG(j,j1+1/2,j2+1/2,m,m1-1/2,m2-1/2)/(2*j2+2)
        rest = sqrt(j1+1+m1p)*sqrt(j2+1+m2p)*CG(j,j1+1/2,j2+1/2,mp,m1p+1/2,m2p+1/2) + sqrt(j1+1-m1p)*sqrt(j2+1-m2p)*CG(j,j1+1/2,j2+1/2,mp,m1p-1/2,m2p-1/2)
        
    return pref*rest

def diracprojCGnaive(n,j,m,mp):
    '''Computes the Dirac projection P_J f^{j}_{m,mp} P_J with J = n/2.
    
    Index order of increasing: mp -> m -> p -> n, so e.g. 
    [(-1, 0, 0, 0), (-1, 0, 1, 0), (1, 0, 0, -1), (1, 0, 0, 0), (-1, 1/2, -1/2, -1/2), (-1, 1/2, -1/2, 1/2), (-1, 1/2, 1/2, -1/2), (-1, 1/2, 1/2, 1/2), (-1, 1/2, 3/2, -1/2), (-1, 1/2, 3/2, 1/2), (1, 1/2, -1/2, -3/2), (1, 1/2, -1/2, -1/2), (1, 1/2, -1/2, 1/2), (1, 1/2, 1/2, -3/2), (1, 1/2, 1/2, -1/2), (1, 1/2, 1/2, 1/2)]'''

    indexlist = [(2*q0-1,n1/2,k1-n1/2,k1p-n1/2) for n1 in range(0,n+1) for q0 in range(0,2) for k1 in range(0,n1+2-q0) for k1p in range(-q0,n1+1)]

    # Create matrix using individual inner products
    bigmat = [[spinorinp(j,m,mp,t[0],t[1],t[2],t[3],r[0],r[1],r[2],r[3]) for t in indexlist] for r in indexlist]
    return np.array(bigmat,dtype=float)

def matlistmaker(n):
    '''Creates the basis of S_J where J = n/2. This is equal to all truncated matrices [[j, m, m']] for j <= 2J + 1/2.'''
    m = 2*n + 1 
    ll = [diracprojCGnaive(n,p/2,q-p/2,r-p/2) for p in range(m+1) for q in range(p+1) for r in range(p+1)]
    return ll

def span_dimension(vectors):
    '''Computes the dimension of the span for a list of 1d arrays (vectors). This is equal to the rank of the matrix with these vectors as rows.'''

    matrix = np.array(vectors, dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    return rank

### For Peter-Weyl prop nr = 2 or 3, so we were content only multiplying all matrices once and seeing if this spans everything.
# For Dirac, prop nr = 4 (computationally), so that we have to multiply more matrices.
# Naively storing a list of all multiples of four basis matrices would be O(NN^12), as the amount of basis matrices is O(NN^3). This is already not feasible for NN = 2 and runs into memory problems. Note that the dimension of C*_{env} is O(NN^6).

# Since I do not have a smart way of choosing what matrices to multiply, I instead opt to multiply all matrices, but throw away linearly dependent matrices.
# Checking if every new matrix is linearly dependent is computationally expensive. As such, we employ computationally cheaper, but less sophisticated methods.
# i.e. throwing away random matrices.
# Nevertheless, I do not want to throw away vital information, so after choosing random matrices to throw out, a check is done which sees if the dimension is preserved.
# In the algorithm, this process occurs every 10000 multiplications.

def randomthrow(listofmats,proportion=4,msg=1):
    '''Given a list of matrices, returns a list of matrices with size equal to proportion*dim(span(listofmats)) with the same dimension of the span. This is achieved by choosing a random subset of matrices and checking the dimension of the span, until the same dimension is achieved.
    This heuristically works around 50% of the time for the use cases involved. Prints intermediately to inform at what step in the loop the program is. Loops for an infinite amount of time with probability zero.'''

    vectlist = [x.flatten() for x in listofmats]
    dim = span_dimension(vectlist)
    matspicked = proportion*dim

    while(True):
        print(f'Throw {msg}')
        shuffle(listofmats)
        randomlist = listofmats[0:matspicked]
        randomlistvects = [x.flatten() for x in randomlist]
        newdim = span_dimension(randomlistvects)
        if dim == newdim:
            break
    return randomlist

def randomprop(listofmats, n, k=8):
    '''Returns the propagation number of matrices in listofmats needed to get all n by n matrices. The maximal amount of propagation can be set with k.
    Intermediately, prints the dimensions of propagated systems, and also prints since randomthrow is called.'''

    print(f'- Dimension of C*env = {n*n}')

    zeromat = np.zeros(dtype=float,shape=(n,n))
    listofmatprods = listofmats.copy()
    print(f'- prop nr 1: dim = {len(listofmats)}')
    listofnewmatprods = []
    propnr = 1
    while(True):
        rank=0
        propnr+=1
        for x in listofmatprods:
            for y in listofmats:
                prod = x @ y
                #perform check to see if matrix is not zero
                if not np.array_equal(prod,zeromat):
                    listofnewmatprods += [prod]
                    rank +=1
                    #throw away superfluous matrices every 10000 multiplications
                    if rank%10000==0:
                        listofnewmatprods = randomthrow(listofnewmatprods,msg=rank//10000)
        vects = [x.flatten() for x in listofnewmatprods]
        dimmy = span_dimension(vects)
        print(f'- prop nr {propnr}: dim = {dimmy}')

        #Cleanup for next step
        listofmatprods = listofnewmatprods
        listofnewmatprods = []
        vects = []
        if dimmy == n*n:
            break
        if propnr==k:
            break
        
        #Reduce amount of superfluous matrices multiplied in next step
        listofmatprods = randomthrow(listofmatprods, proportion=3, msg='end of prop nr loop')
    return propnr


def randompropSJ(NN,k=8):
    basismats = matlistmaker(NN)
    matlength = 2*(NN+1)*(NN+2)*(NN+3)//3
    randomprop(basismats, matlength,k)



propnr = randompropSJ(NN)
