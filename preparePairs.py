import os
import sys
import numpy as np

#Getting command line input
inputFile1 = sys.argv[1]
numPosPairs = int(sys.argv[2])
inputFile2 = sys.argv[3]
numNegPairs = int(sys.argv[4])
requiredVectorsFile = sys.argv[5]
stopWordsFile = sys.argv[6]
outputFile = sys.argv[7]

#Loading vector keys
print ('Loading vector keys')
vectors = {}
vecFile = open(requiredVectorsFile)
for line in vecFile:
	lineParts = line.split()
	vectors[lineParts[0]] = ''

#Reading stop words for processing queries
print ('Reading stop words for processing queries')
stopWords = {}
f = open(stopWordsFile)
for line in f:
	stopWords[line.strip()] = ''
f.close()

#Opening the pairs file
fw = open(outputFile, 'w')

#Reading the positive pairs input file and adding to the pairs file to be input to NN code
print ('Reading positive pairs')
iterCount = 0
f = open(inputFile1)
for line in f:
	iterCount += 1
	if(iterCount % 10000 == 0):
		print '.',
	
	if(iterCount > numPosPairs):
		break
	
	lineParts = line.split('||')
	tq = lineParts[0].strip()
	hq = lineParts[1].strip()
	
	#Processing stopWords and creating intermediate query
	newTq = ''
	for word in tq.split():
		if(word in stopWords):
			continue
		else:
			newTq = newTq + ' ' + word
	newTq = newTq.strip()
	if(len(newTq.strip()) == 0):
		iterCount -= 1
		continue

	newHq = ''
	for word in hq.split():
		if(word in stopWords):
			continue
		else:
			newHq = newHq + ' ' + word
	newHq = newHq.strip()
	if(len(newHq.strip()) == 0):
		iterCount -= 1
		continue

	#Modifying queries to their 10 word (tail query) and 5 word (head query) versions 
	#Also replacing missing/non-existent words with special term ~ZERO~
	tq = ''
	hq = ''

	tqLen = len(newTq.split())
	if(tqLen > 10):
		i = 0
		for word in newTq.split():
			i = i + 1
			if(i > 10):
				break
			
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
	else:
		for word in newTq.split():
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
		
		for i in range(10 - tqLen):
			tq = tq + ' ~ZERO~'
	tq = tq.strip()
	
	hqLen = len(newHq.split())
	if(hqLen > 5):
		i = 0
		for word in newHq.split():
			i = i + 1
			if(i > 5):
				break
			
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
	else:
		for word in newHq.split():
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
		
		for i in range(5 - hqLen):
			hq = hq + ' ~ZERO~'
	hq = hq.strip()
		
	fw.write(tq + '\t' + hq + '\t1\n')

#Reading the negative pairs input file and adding to the pairs file
print ('')
print ('Reading negative pairs')
iterCount = 0
f = open(inputFile2)
for line in f:
	iterCount += 1
	if(iterCount % 100000 == 0):
		print '.',
	
	if(iterCount > numNegPairs):
		break
	
	lineParts = line.split('||')
	tq = lineParts[0].strip()
	hq = lineParts[1].strip()
	
	#Processing stopWords and creating intermediate query
	newTq = ''
	for word in tq.split():
		if(word in stopWords):
			continue
		else:
			newTq = newTq + ' ' + word
	newTq = newTq.strip()
	if(len(newTq.strip()) == 0):
		iterCount -= 1
		continue

	newHq = ''
	for word in hq.split():
		if(word in stopWords):
			continue
		else:
			newHq = newHq + ' ' + word
	newHq = newHq.strip()
	if(len(newHq.strip()) == 0):
		iterCount -= 1
		continue

	#Modifying queries to their 10 word (tail query) and 5 word (head query) versions 
	#Also replacing missing/non-existent words with special term ~ZERO~
	tq = ''
	hq = ''

	tqLen = len(newTq.split())
	if(tqLen > 10):
		i = 0
		for word in newTq.split():
			i = i + 1
			if(i > 10):
				break
			
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
	else:
		for word in newTq.split():
			if(word in vectors):
				tq = tq + ' ' + word
			else:
				tq = tq + ' ~ZERO~'
		
		for i in range(10 - tqLen):
			tq = tq + ' ~ZERO~'
	tq = tq.strip()
	
	hqLen = len(newHq.split())
	if(hqLen > 5):
		i = 0
		for word in newHq.split():
			i = i + 1
			if(i > 5):
				break
			
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
	else:
		for word in newHq.split():
			if(word in vectors):
				hq = hq + ' ' + word
			else:
				hq = hq + ' ~ZERO~'
		
		for i in range(5 - hqLen):
			hq = hq + ' ~ZERO~'
	hq = hq.strip()
		
	fw.write(tq + '\t' + hq + '\t0\n')
fw.close()
