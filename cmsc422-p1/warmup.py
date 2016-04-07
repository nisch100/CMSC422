import util, datasets, binary, dumbClassifiers, runClassifier
from numpy import * 
from pylab import *

print 'AlwaysPredictOne: '
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.TennisData)
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.GenderData)
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.SentimentData)

print '\nAlwaysPredictMostFrequent: '
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.TennisData)
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.GenderData)
runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.SentimentData)

print '\nFirstFeatureClassifer: '
runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData)
runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.GenderData)
runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.SentimentData)
