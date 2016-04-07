import util, datasets, binary, dumbClassifiers, runClassifier
from numpy import * 
from pylab import *

print 'AlwaysPredictOne: '
h = dumbClassifiers.AlwaysPredictOne({})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
h.predictAll(datasets.TennisData.X)
runClassifier.trainTestSet(h, datasets.TennisData)

print '\nAlwaysPredictMostFrequent: '
h = dumbClassifiers.AlwaysPredictMostFrequent({})
runClassifier.trainTestSet(h, datasets.TennisData)

print '\nFirstFeatureClassifer: '
h = dumbClassifiers.FirstFeatureClassifier({})
runClassifier.trainTestSet(h, datasets.TennisData)
