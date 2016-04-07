import util, datasets, binary, dumbClassifiers, runClassifier, dt, knn
from numpy import * 
from pylab import *

print "Quotes Data"
runClassifier.trainTestSet(dt.DT({'maxDepth': 32}), datasets.QuotesData)
#runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.QuotesData)
#runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.QuotesData)

