#THMapNet_v2 - New network architecture which involves use of embeddings of DBPedia entities
import sys
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from keras.models import Model
from keras.layers import Input, merge, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten

trainFile = sys.argv[1]
vectorFile = sys.argv[2]
vectorDim = int(sys.argv[3])
entityVectorFile = sys.argv[4]
queryEntityMappingFile = sys.argv[5]
epos = int(sys.argv[6])

#Loading word embeddings
print ('Loading Word Embeddings')
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
vecFile.close()

#Loading entity embeddings
print ('Loading Entity Embeddings')
entVectors = {}
entVecFile = open(entityVectorFile)
for line in entVecFile:
	if(line.strip().find(' ') < 0):
		continue
	lineParts = line.split()
	temp = np.asarray(lineParts[1:])
	try:
		vector = temp.astype(np.float)
		entVectors[lineParts[0]] = vector
	except:
		print 'Not loading: ' + lineParts[0]
		pass
entVecFile.close()

#Loading query entity mappings
print ('Loading Query Entity Mappings')
queryEntityMap = {}
queryEntMapFile = open(queryEntityMappingFile)
for line in queryEntMapFile:
	if(line.strip().find('\t') < 0):
		continue
	lineParts = line.split('\t')
	if(len(lineParts) >= 4):
		queryEntityMap[lineParts[0]] = lineParts[1:4]
	else:
		queryEntityMap[lineParts[0]] = lineParts[1:]

#Reading training file
print ('Loading Train Data')
trFile = open(trainFile)
X = []
for line in trFile:
	line = line.decode('utf-8')
	lineParts = line.split('\t')
	X.append((lineParts[0], lineParts[1], lineParts[2]))
trFile.close()
print str(len(X))

#Shuffling the input so that positive and negative examples are disarranged
random.shuffle(X)

#Splitting into training and test sets (80-20 split)
splitTrainEnd = int(math.ceil(0.8 * len(X)))
splitTestEnd = len(X)

zeroVec = np.zeros((vectorDim,))
X_train_tq = np.empty((splitTrainEnd, 10, vectorDim))
X_train_tq_ent = np.empty((splitTrainEnd, 1500))
X_train_hq = np.empty((splitTrainEnd, 5, vectorDim))
X_train_hq_ent = np.empty((splitTrainEnd, 500))
Y_train = np.empty((splitTrainEnd, 1))
pos = 0
neg = 0
for i in range(splitTrainEnd):
	tq = X[i][0]
	innerArr = np.empty((10, vectorDim))
	tqParts = tq.split(' ')
	for j in range(len(tqParts)):
		if(tqParts[j] == '~ZERO~' or tqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[tqParts[j]]
	X_train_tq[i] = innerArr
	
	entVec = np.zeros(1500)
	if(tq in queryEntityMap):
		entitiesTq = queryEntityMap[tq]
		i = 0
		for entityTq in entitiesTq:
			currI = i * 500
			if(entityTq in entVectors):
				entVec[currI:currI+500] = entVectors[entityTq]
			i += 1
	X_train_tq_ent[i] = entVec
	
	hq = X[i][1]
	innerArr = np.empty((5, vectorDim))
	hqParts = hq.split() 
	for j in range(len(hqParts)):
		if(hqParts[j] == '~ZERO~' or hqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[hqParts[j]]
	X_train_hq[i] = innerArr
	
	entVec = np.zeros(500)
	if(hq in queryEntityMap):
		entitiesHq = queryEntityMap[hq]
		entityHq = entitiesHq[0]
		if(entityHq in entVectors):
			entVec = entVectors[entityHq]
	X_train_hq_ent[i] = entVec
	
	Y_train[i] = float(X[i][2])
	if(Y_train[i] == 1.0):
		pos += 1
	else:
		neg += 1
print ('Training - Positive: ' + str(pos) + ' Negative: ' + str(neg))

print (np.where(np.isfinite(X_train_tq) == False))
print (np.where(np.isfinite(X_train_tq_ent)== False))
print (np.where(np.isfinite(X_train_hq)== False))
print (np.where(np.isfinite(X_train_hq_ent) == False))

numTestInst = splitTestEnd - splitTrainEnd
X_test_tq = np.empty((numTestInst, 10, vectorDim))
X_test_tq_ent = np.empty((numTestInst, 1500))
X_test_hq = np.empty((numTestInst, 5, vectorDim))
X_test_hq_ent = np.empty((numTestInst, 500))
Y_test = np.empty((numTestInst, 1))
pos = 0
neg = 0
for i in range(numTestInst):
	k = i + splitTrainEnd
	tq = X[k][0]
	innerArr = np.empty((10, vectorDim))
	tqParts = tq.split(' ')
	for j in range(len(tqParts)):
		if(tqParts[j] == '~ZERO~' or tqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[tqParts[j]]
	X_test_tq[i] = innerArr
	
	entVec = np.zeros(1500)
	if(tq in queryEntityMap):
		entitiesTq = queryEntityMap[tq]
		i = 0
		for entityTq in entitiesTq:
			currI = i * 500
			if(entityTq in entVectors):
				entVec[currI:currI+500] = entVectors[entityTq]
			i += 1
	X_test_tq_ent[i] = entVec
	
	hq = X[k][1]
	innerArr = np.empty((5, vectorDim))
	hqParts = hq.split(' ') 
	for j in range(len(hqParts)):
		if(hqParts[j] == '~ZERO~' or hqParts[j] not in vectors):
			innerArr[j] = zeroVec
		else:
			innerArr[j] = vectors[hqParts[j]]
	X_test_hq[i] = innerArr

	entVec = np.zeros(500)
	if(hq in queryEntityMap):
		entitiesHq = queryEntityMap[hq]
		entityHq = entitiesHq[0]
		if(entityHq in entVectors):
			entVec = entVectors[entityHq]
	X_test_hq_ent[i] = entVec

	Y_test[i] = float(X[k][2])
	if(Y_test[i] == 1.0):
		pos += 1
	else:
		neg += 1
print ('Testing - Positive: ' + str(pos) + ' Negative: ' + str(neg))

print ('Data sizes:')
print str(X_train_hq.shape) + '\t' + str(Y_train.shape) + '\t' + str(X_test_hq.shape) + '\t' + str(Y_test.shape)

batchSize = 32

print ('Creating NN Graph model')
tqInp = Input(shape=(10, vectorDim), name='tqInput')
tqInpEnt = Input(shape=(1500,), name='tqInputEnt')
hqInp = Input(shape=(5, vectorDim), name='hqInput')
hqInpEnt = Input(shape=(500,), name='hqInputEnt')

tqLSTM = LSTM(300)(tqInp)
hqLSTM = LSTM(150)(hqInp)

"""merge1 = merge([tqLSTM, tqInpEnt], mode='concat')
merge2 = merge([hqLSTM, hqInpEnt], mode='concat')
merge3 = merge([merge1, merge2], mode='concat')
hidden1 = Dense(600, activation='sigmoid')(merge3)"""

merge1 = merge([tqLSTM, tqInpEnt, hqLSTM, hqInpEnt], mode='concat')
hidden1 = Dense(600, activation='relu')(merge1)

out = Dense(1, activation='sigmoid')(hidden1)

model = Model([tqInp, tqInpEnt, hqInp, hqInpEnt], out)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.summary()

#Training model
print ('Training')
model.fit([X_train_tq, X_train_tq_ent, X_train_hq, X_train_hq_ent], [Y_train], nb_epoch = epos, batch_size = batchSize)

#Testing
print ('Testing')
Y_pred = model.predict({'tqInput': X_test_tq, 'tqInputEnt': X_test_tq_ent, 'hqInput': X_test_hq, 'hqInputEnt': X_test_hq_ent})
Y_pred_final = np.empty(Y_pred.shape)
for i in range(len(Y_pred)):
	if(Y_pred[i] > 0.5):
		Y_pred_final[i] = 1
	else:
		Y_pred_final[i] = 0

Y_test_final = np.empty(Y_test.shape)
for i in range(len(Y_test)):
	if(Y_test[i] == 1.0):
		Y_test_final[i] = 1
	else:
		Y_test_final[i] = 0

c = confusion_matrix(Y_test_final, Y_pred_final)
print c
precision, recall, f1, support = precision_recall_fscore_support(Y_test_final, Y_pred_final)
print('Precision: ' + str(precision) + '\tRecall: ' + str(recall) + '\tF1: ' + str(f1))
