from multiclass import *
from binary import *
from dt import *
from datasets import *
from util import *


print WineData.labels[17]
h = OVA(3, lambda: DT({'maxDepth': 20}))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)

print h.f[0].displayTree(0, WineData.words)

#print "smaller data set"#

#h = OVA(5, lambda: DT({'maxDepth': 3}))
#h.train(WineData.X, WineData.Y)
#P = h.predictAll(WineData.Xte)
#print mean(P == WineData.Yte)
#freqCl = mode(WineData.Y)
#print freqCl
#print WineData.labels[freqCl]
#print mean(WineData.Yte == freqCl)#

#WineDataSmall.labels[0]
#print h.f[0].displayTree(0, WineDataSmall.words)#

#print "hkfds"#

#WineDataSmall.labels[2]
#print h.f[2].displayTree(0, WineDataSmall.words)
