import util, datasets, binary, dumbClassifiers, runClassifier, knn
from numpy import * 
from pylab import *

#print 'Generating epsilon ball curve'
#curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN': False}), 'eps', [2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.5], datasets.DigitData)
#runClassifier.plotCurve('Epsilon ball on Digits Data (hyperparameter)', curve)

#curve = runClassifier.hyperparamCurveSet(knn.KNN({'isKNN': True}), 'K', [1, 5, 10, 50, 75, 99], datasets.DigitData)
#runClassifier.plotCurve('KNN on Digits Data (hyperparameter)', curve)

curve = runClassifier.learningCurveSet(knn.KNN({'isKNN': True, 'K': 5}), datasets.DigitData)
runClassifier.plotCurve('Learning Curve on Digits Data (K=5)', curve)
