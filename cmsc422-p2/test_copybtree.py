from multiclass import *
from datasets import *
from dt import *

treeP = {}
avaP = {}
ovaP = {}
for i in range(1,20)
    t = MCTree(getMyTreeForWine(), lambda: DT({'maxDepth': i}))
    t.train(WineData.X, WineData.Y)
    tP = t.predictAll(WineData.Xte)
    treeP[i] = mean(tP == WineData.Yte)
    
    A = AVA(20, lambda: DT({'maxDepth': i}))
    A.train(WineData.X, WineData.Y)
    aP = A.predictAll(WineData.Xte)
    avaP[i] = mean(aP == WineData.Yte)

    O = OVA(20, lambda: DT({'maxDepth': i}))
    O.train(WineData.X, WineData.Y)
    oP = O.predictAll(WineData.Xte)
    ovaP[i] = mean(oP == WineData.Yte)
    


print "Tree: " + str(treeP)
print "AVA: " + str(avaP)
print "OVA: " + str(ovaP)

