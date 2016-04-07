import util, datasets, binary, dumbClassifiers, runClassifier, dt
from numpy import * 
from pylab import *

print 'Tennis MaxDepth 1: '
h = dt.DT({'maxDepth': 1})
print h
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h

print '\n\nTennis MaxDepth 2: '
h = dt.DT({'maxDepth': 2})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h

print '\n\nTennis MaxDepth 5: '
h = dt.DT({'maxDepth': 5})
h.train(datasets.TennisData.X, datasets.TennisData.Y)
print h

'''
print '\n\nGender MaxDepth 2: '
h = dt.DT({'maxDepth': 2})
h.train(datasets.GenderData.X, datasets.GenderData.Y)
print h

print '\n\nPredict Gender: '
runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.GenderData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.GenderData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.GenderData)

print '\n\nPredict Sentiment: '
runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.SentimentData)
runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.SentimentData)

print '\n\nGender Learning Curve: '
curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 9}), datasets.GenderData)
runClassifier.plotCurve('DT on Gender Data (hyperparameter)', curve)

print '\n\nGender Data Hyperparameter: '
curve = runClassifier.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,8,16,32], datasets.GenderData)
runClassifier.plotCurve('DT on Gender Data (hyperparameter)', curve)
'''
