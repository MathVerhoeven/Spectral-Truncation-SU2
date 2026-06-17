import sympy as sp
from sympy.physics.wigner import clebsch_gordan as sp_cg
import numpy as np
from numpy import sqrt as sqrt #used to be from math
import matplotlib.pyplot as plt

#Plot the Spectral Localizer (SL) as a function of kappa for several truncation parameters.

#Here j = n/2 is the label of the irrep \pi_j which is analyzed by the SL
#The lowest truncation level plotted is J_{min} = mintruncation/2, the highest is J_{max} = maxtruncation/2

n = 1
mintruncation = 0
maxtruncation = 4

########################################

maxtruncation = maxtruncation + 1 #for later use in range(mintruncation,maxtruncation)

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

def diracdiag(trunc):
    '''Returns the diagonalized Dirac operator on SU(2). Based on the decomposition E_{-0} + E_{0} + E_{-1} + E_{1} + ... consistent with the ordering provided by diracprojCGnaive.''' 
    eigvals = []
    for i in range(trunc+1):
        #eigenvalue repeated amount of times equal to dimension of eigenspace
        eigvals+=[-(3/2+i)]*((i+1)*(i+2))
        eigvals+=[(3/2+i)]*((i+1)*(i+2))
    return np.diag(eigvals)

def oplus(A,B):
    '''Forms the direct sum of two square 2d numpy arrays.'''
    #assumes both A and B are square matrices
    Alen =  A.shape[0]
    Blen = B.shape[0]
    zeromat = np.zeros((Alen,Blen))
    dirsum = np.block([[A,zeromat],[np.transpose(zeromat),B]])
    return dirsum

def diracN(trunc,n):
    '''Returns the truncated Dirac operator repeated n times along the diagonal. That is, ([[D]]_trunc)^{\oplus n}.'''
    dirac = diracdiag(trunc)
    res = dirac
    for i in range(n-1):
        res = oplus(res,dirac)
    return res

def irreptrunc(trunc,n):
    '''Returns the truncated irrep \pi_{n/2} viewed as a matrix of elements in S_trunc. That is, [[\pi_{n/2}]]_trunc'''
    #Get block matrix of L2-normalized matrix coefficients
    normalizedmat = [[diracprojCGnaive(trunc,n/2,-i+n/2,-j+n/2) for i in range(n+1)] for j in range(n+1)]
    #`Unnormalize' the coefficients to get the original representation back
    mat = sqrt(1/(n+1))*np.block(normalizedmat)
    return mat


def sigSLplot(trunc, n, show=False):
    '''Plots half the signature of the spectral localizer L_{\kappa, trunc}(\pi_{n/2}) as a function of \kappa between 0 and 1.'''
    irrep = irreptrunc(trunc,n)
    irreptrans = np.transpose(irrep) #= hermitian tranpose, every inner product is real
    diracmat = diracN(trunc,n+1) #n+1 due to dimension of V_n

    #Define Spectral Localizer with tuning parameter kappa
    def SpecLoc(kappa):
        SLmat = np.block([[kappa*diracmat, irrep],[irreptrans, -kappa*diracmat]])
        return SLmat

    #Define signature of Spectral Localizer
    def sigSL(kappa):
        SLmat = np.array(SpecLoc(kappa),dtype=float)
        eigvals = np.linalg.eigvalsh(SLmat) #eigvalsh used for symmetric matrix
        pos = np.sum(eigvals > 0)
        neg = np.sum(eigvals < 0)
        return pos - neg    
    
    # Apply sigSL for many values of kappa
    npsigSL = np.vectorize(sigSL)
    xvals = np.linspace(0+1/400,1,400)
    yvals = 0.5*npsigSL(xvals)

    plt.plot(xvals, yvals)
    if show:
        plt.xlabel("kappa")
        plt.ylabel("Half signature")
        plt.title("Plot of half signature")
        plt.show()

## Plot the half-signature of the spectral localizer

for i in range(mintruncation,maxtruncation):
    sigSLplot(i,n)

plt.xlabel("kappa")
plt.ylabel("Half signature")
if n%2 == 0:
    plt.title("Plot of half signature, j = "+str(n//2))
else:
    plt.title("Plot of half signature, j = "+str(n)+'/2')

#Rescale plot to not show y > 1, as this part is not relevant for these irreps
ymin, ymax = plt.ylim()
plt.ylim(ymin, 1)
plt.legend([f'rho = {2*i+3}/2' for i in range(mintruncation,maxtruncation)])


plt.show()

