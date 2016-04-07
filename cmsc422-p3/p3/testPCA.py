from numpy import *
from numpy.random import *
from pylab import *
import util, dr, datasets, kernel

Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(randn(1000,2), Si)
plot(x[:,0], x[:,1], 'b.')
show()
print dot(x.T,x) / real(x.shape[0])
(P,Z,evals) = dr.pca(x, 2)
print Z
print evals
x0 = dot(dot(x, Z[0,:]).reshape(1000,1), Z[0,:].reshape(1,2))
x1 = dot(dot(x, Z[1,:]).reshape(1000,1), Z[1,:].reshape(1,2))
plot(x[:,0], x[:,1], 'b.', x0[:,0], x0[:,1], 'r.', x1[:,0], x1[:,1], 'g.')
show()
(X,Y) = datasets.loadDigits()
(P,Z,evals) = dr.pca(X, 784)
print evals
print "Cov:"
normeval = evals/sum(evals)
plot(normeval)
show()
cumul = cumsum(normeval)

#for x in range(0, len(cumul)):
#    print x, cumul[x]

print 'WU3'
util.drawDigits(Z.T[0:50,:], arange(50))
show()
print 'WU4'
Si = util.sqrtm(array([[3,2],[2,4]]))
x = dot(randn(1000,2), Si)
(P, alpha, evals) = dr.kpca(X, 2, kernel.linear)
print "eigenvalues:"
print evals
print "alpha:"
print alpha
(a,b) = datasets.makeKPCAdata()
plot(a[:,0], a[:,1], 'b.', b[:,0], b[:,1], 'r.')
show()
exit()
x = vstack((a,b))
(P,Z,evals) = dr.pca(x, 2)
print 'PCA evecs: ' + str(Z)
print 'PCA evals: ' + str(evals)

print 'WU5'
Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:-1,:]
plot(Pa[:,0], randn(Pa.shape[0]), 'b.', Pb[:,0], randn(Pb.shape[0]), 'r.')
show()
(P,alpha,evals) = dr.kpca(x, 2, kernel.rbf1)
print 'kpca evals: ' + str(evals)
print 'alphas: ' + str(alpha)
print 'WU6'
print 'kpca plot'
Pa = P[0:a.shape[0],:]
Pb = P[a.shape[0]:-1,:]
plot(Pa[:,0], Pa[:,1], 'b.', Pb[:,0], Pb[:,1], 'r.')
show()
