from multiclass import *
from datasets import *
from dt import *

h = AVA(5, lambda: DT({'maxDepth': 1}))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
print mean(P == WineDataSmall.Yte)

print ""

h = AVA(5, lambda: DT({'maxDepth': 3}))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
print mean(P == WineDataSmall.Yte)

print h.f[2][1].displayTree(0, WineDataSmall.words)
