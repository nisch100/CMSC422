from multiclass import *
from datasets import *
from dt import *

h = MCTree(getMyTreeForWine(), lambda: DT({'maxDepth': 5}))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)
print mean(P == WineData.Yte)
