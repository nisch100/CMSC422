import util, datasets, binary, dumbClassifiers, runClassifier, knn
from numpy import * 
from pylab import *

print 'Tennis Data Epsillon: '
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 0.5}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 1.0}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 2.0}), datasets.TennisData)

'''
print 'Tennis Data K: '
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.TennisData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.TennisData)
'''

print 'Digit Data Epsillon: '
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 6.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 8.0}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': False, 'eps': 10.0}), datasets.DigitData)

'''
print '\n\nDigit Data K: '
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 1}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 3}), datasets.DigitData)
runClassifier.trainTestSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)

'''
