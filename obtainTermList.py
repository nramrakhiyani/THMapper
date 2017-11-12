import os
import sys
import numpy as np

#Getting command line input
inputFile1 = sys.argv[1]
inputFile2 = sys.argv[2]
stopWordFile = sys.argv[3]
outputFile = sys.argv[4]

#Reading stop words for processing queries
stopWords = {}
f = open(stopWordFile)
for line in f:
	stopWords[line.strip()] = ''
f.close()

#Reading input file 1 and preparing term list
iterCount = 0
termList = {}
print ('Reading Input file 1')
f = open(inputFile1)
for line in f:
	iterCount += 1
	if(iterCount % 10000 == 0):
		print '.',

	lineParts = line.split('||')
	tq = lineParts[0].strip()
	hq = lineParts[1].strip()
	
	#Processing stopWords and collecting termlist
	for word in tq.split():
		if(word in stopWords):
			continue
		else:
			if(word not in termList):
				termList[word] = ''

	for word in hq.split():
		if(word in stopWords):
			continue
		else:
			if(word not in termList):
				termList[word] = ''
f.close()

#Reading input file 2 and adding to the term list
iterCount = 0
print ('')
print ('Reading Input file 2')
f = open(inputFile2)
for line in f:
	iterCount += 1
	if(iterCount % 10000 == 0):
		print '.',

	lineParts = line.split('||')
	tq = lineParts[0].strip()
	hq = lineParts[1].strip()
	
	#Processing stopWords and collecting termlist
	for word in tq.split():
		if(word in stopWords):
			continue
		else:
			if(word not in termList):
				termList[word] = ''

	for word in hq.split():
		if(word in stopWords):
			continue
		else:
			if(word not in termList):
				termList[word] = ''
f.close()

#Print the termList to get vectors from the main file
f = open(outputFile, 'w')
for term in termList:
	f.write(term.strip() + '\n')
f.close()
