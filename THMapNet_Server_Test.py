import sys
import math
import heapq
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.models import Model, load_model
from keras.layers import Input, merge, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten

tqhqMappingFilePath = sys.argv[1]
vectorFile = sys.argv[2]
vectorDim = int(sys.argv[3])
modelFile = sys.argv[4]
outputFilePath = sys.argv[5]

#Loading Tail Queries and Tail Query-Head Query mapping
print ('Loading Tail Query-Head Query mapping')
tailQueries = []
tqhqMap = {}
tqhqMappingFile = open(tqhqMappingFilePath)
for line in tqhqMappingFile:
	lineParts = line.split('||')
	tq = lineParts[0].strip()
	tailQueries.append(tq)
	headQueries = []
	for hq in lineParts[1:]:
		currHq = hq.strip()
		currHq = currHq[0:currHq.rfind(':')]
		headQueries.append(currHq)
	tqhqMap[tq] = headQueries
tqhqMappingFile.close()

#Loading vectors
print ('Loading Vectors')
vectors = {}
vecFile = open(vectorFile)
for line in vecFile:
	lineParts = line.split()
	temp = np.asarray(lineParts[1:])
	try:
		vector = temp.astype(np.float)
		vectors[lineParts[0]] = vector
	except:
		print 'Not loading: ' + lineParts[0]
		pass

#Loading Learned Neural Model
print ('Loading Learned Neural Model')
model = load_model(modelFile)

def getPairFormat(q, n):
	newQ = ''
	qLen = len(q.split())
	if(qLen > n):
		i = 0
		for word in q.split():
			i = i + 1
			if(i > n):
				break
			
			if(word in vectors):
				newQ = newQ + ' ' + word
			else:
				newQ = newQ + ' ~ZERO~'
	else:
		for word in q.split():
			if(word in vectors):
				newQ = newQ + ' ' + word
			else:
				newQ = newQ + ' ~ZERO~'
		
		for i in range(n - qLen):
			newQ = newQ + ' ~ZERO~'
	newQ = newQ.strip()
	return newQ

def getBestHeadQueries(tq):
	zeroVec = np.zeros((vectorDim,))
	modifiedTq = getPairFormat(tq, 10)
	tqArr = np.empty((10, vectorDim))
	tqParts = modifiedTq.split(' ')
	for j in range(len(tqParts)):
		if(tqParts[j] == '~ZERO~' or tqParts[j] not in vectors):
			tqArr[j] = zeroVec
		else:
			tqArr[j] = vectors[tqParts[j]]
	
	headQueries = tqhqMap[tq]
	testDataSize = len(headQueries)
	X_test_tq = np.empty((testDataSize, 10, vectorDim))
	X_test_hq = np.empty((testDataSize, 5, vectorDim))
	for i in range(testDataSize):
		hq = headQueries[i]
		modifiedHq = getPairFormat(hq, 5)
		innerArr = np.empty((5, vectorDim))
		hqParts = modifiedHq.split()
		for j in range(len(hqParts)):
			if(hqParts[j] == '~ZERO~' or hqParts[j] not in vectors):
				innerArr[j] = zeroVec
			else:
				innerArr[j] = vectors[hqParts[j]]
		X_test_hq[i] = innerArr
		X_test_tq[i] = tqArr

	Y_pred = model.predict({'tqInput': X_test_tq, 'hqInput':X_test_hq})
	hqScore = {}
	for i in range(testDataSize):
		#if(Y_pred[i] >= 0.5):
		hqScore[headQueries[i]] = Y_pred[i]
	
	if(len(hqScore) > 0):
		bestHQs = heapq.nlargest(5, hqScore, key=hqScore.get)
	else:
		bestHQs = {}
	return bestHQs

#Trying for each tail query in the test set
outputFile = open(outputFilePath, 'w')
for tailQuery in tailQueries:
	print 'Checking for: ' + tailQuery
	bestHQList = getBestHeadQueries(tailQuery)
	if(len(bestHQList) > 0):
		outputFile.write(tailQuery + '\t' + '\t'.join(bestHQList) + '\n')
outputFile.close()
