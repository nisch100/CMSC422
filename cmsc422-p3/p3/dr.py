from numpy import *
from util import *
from pylab import *


def pca(X, K):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})

    should return a tuple (P, Z, evals)

    where P is the projected data (N*K) where
    the first dimension is the higest variance,
    the second dimension is the second higest variance, etc.

    Z is the projection matrix (D*K) that projects the data into
    the low dimensional space (i.e., P = X * Z).

    and evals, a K dimensional array of eigenvalues (sorted)
    '''

    N, D = X.shape

    '''
    SHOULD BE 5 LINE FUNCTIONS IF WE DO IT right
    USE NUMPY EIGENVALUE FUNCTIONS!

    Notes on Correlation from STAT400:
    Correlation(X, Y) = E[(X - mu_x)(Y - mu_y)] = E(XY) - mu_x*mu_y

    Notes on Covariance from STAT400:
    Covariance(X, Y) = Correlation(X, Y) / (sigma_x * sigma_y),   possible values range [-1, 1],   sigma is standard deviation
    Covariance measures how linearly correlated two things are
    If Cov(x, y) = -1 or 1, then the data is linearlly correlated. That means Y is a linear combination of X, or Y=aX+b.
    If Cov(x, y) = 0, then the X data is independent of the Y data completely. No linear correlation.
    Else, the data is some amount linearly correlated.
    It's kind of like the dot product of two sets of data.
    '''

    # make sure we don't look for too many eigs!
    if K > N:
        K = N
    if K > D:
        K = D

    # first, we need to center the data
    ### TODO: YOUR CODE HERE
    X = X - np.mean(X, axis=0)

    # next, compute eigenvalues of the data variance
    #    hint 1: look at 'help(pylab.eig)'
    #    hint 2: you'll want to get rid of the imaginary portion of the eigenvalues; use: real(evals), real(evecs)
    #    hint 3: be sure to sort the eigen(vectors,values) by the eigenvalues: see 'argsort', and be sure to sort in the right direction!
    #
    ### TODO: YOUR CODE HERE
    #Set rowvar to false so that each column represents a dimension(variable)
    D = np.cov(X.T)
    evals, Z = eig(D)
    evals = real(evals)
    Z = real(Z)
    #[::-1] indexes in reverse = descending order
    
    sortIndices = evals.argsort()[::-1]
    evals = evals[sortIndices]
    Z = Z[:,sortIndices]
    #Only want first K eVals/Vecs
    evals = evals[0:K]
    Z = Z[0:K]
    
    P = np.dot(X, Z) 
    
    return (P, Z, evals)


def kpca(X, K, kernel):
    '''
    X is an N*D matrix of data (N points in D dimensions)
    K is the desired maximum target dimensionality (K <= min{N,D})
    kernel is a FUNCTION that computes K(x,z) for some desired kernel and vectors x,z

    should return a tuple (P, alpha, evals), where P and evals are
    just like in PCA, and alpha is the vector of alphas (mixing
    parameters) for the kernel PCA decomposition.
    '''

    N, D = X.shape

    # first, compute the kernel matrix, called G for gram matrix
    G = zeros((N, N))

    ### TODO: YOUR CODE HERE
    XT = X.T
    for i in range(N):
        for j in range(N):
            G[i][j] = kernel(X[i,:], XT[:,j])
    # next, center the kernel matrix (be careful!!!)
    G0 = G
    oneN = ones((N, N)) / float(N)
    G = G0 - oneN    # TODO: YOUR CODE HERE

    # compute the eigendecomposition of G into evals and evecs
    ### TODO: YOUR CODE HERE
    G = G/N
    evals, evecs = eig(G)
    evals = real(evals)
    evecs = real(evecs)
    # compute the alphas
    alpha = zeros((N, K))
    ### TODO: YOUR CODE HERE
    #[::-1] indexes in reverse = descending order
    sortIndices = evals.argsort()[::-1]
    evals = evals[sortIndices]
    evecs = evecs[:,sortIndices]
    #Only want first K eVals/Vecs
    
    evals = evals[0:K]
    '''
    evecs = evecs[0:K]
    '''
    for i,eval in enumerate(evals):
        alpha[:,i] = evecs[:,i]/sqrt(eval)
        
    # compute the projection of the data points
    P = dot(G, alpha)

    return (P, alpha, evals)
