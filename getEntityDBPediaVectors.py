import os
import sys
import json
import requests
import numpy as np

from gensim.models import Word2Vec

#Getting command line input
inputFilePaths = []
inputFilePaths.append(sys.argv[1])
inputFilePaths.append(sys.argv[2])
linkProbStrength = float(sys.argv[3])
queryToEntityFilePath = sys.argv[4]
entityToVectorFilePath = sys.argv[5]

#Declaring global dictionaries
queryToEntities = {}
entityToVector = {}

#Setting Dexter server parameters
proxy = {"http": None, "https": None}
url = "http://localhost:8080/dexter-webapp/api/rest/annotate"
params = "n=50&wn=true&debug=false&format=text&min-conf=0.5"

#Reading query files and creating query to entity mapping
print ('Reading Input query files and creating query to entity mappings')
for inputFilePath in inputFilePaths:
	iterCount = 0
	print ('Reading ' + inputFilePath)
	inputFile = open(inputFilePath)
	for line in inputFile:
		
		iterCount += 1
		if(iterCount % 1000 == 0):
			print '.',
			sys.stdout.flush()
		if(iterCount % 10000 == 0):
			print iterCount
			sys.stdout.flush()

		lineParts = line.strip().split('||')
		queries = []
		queries.append(lineParts[0].strip())
		queries.append(lineParts[1].strip())
		
		for query in queries:
			if(query not in queryToEntities and len(query.strip()) > 0):
				currEntities = {}
				try:
					finalURL = url + "?text=" + query + "&" + params
					response = requests.get(finalURL, proxies = proxy)
					currDexterJSON = json.loads(response.content, encoding = "utf-8")

					for i in range(len(currDexterJSON['spots'])):
						currEntityLinkProb = float(currDexterJSON['spots'][i]['linkProbability'])
						if(currEntityLinkProb >= linkProbStrength):
							currEntity = currDexterJSON['spots'][i]['wikiname'].strip()
							currEntities[currEntity] = ''
							if(currEntity not in entityToVector):
								entityToVector[currEntity] = ''
				except:
					continue

				queryToEntities[query] = currEntities
	inputFile.close()
	print ''

print ('Writing the query to entities file')
queryToEntityFile = open(queryToEntityFilePath, 'w')
for query in queryToEntities:
	currEntities = queryToEntities[query]
	entityStr = ''
	for currEntity in currEntities:
		entityStr = entityStr + '\t' + currEntity
	try:
		queryToEntityFile.write(query.encode('utf-8').strip() + '\t' + entityStr.encode('utf-8').strip() + '\n')
	except:
		continue
queryToEntityFile.close()

#Reading entities and obtaining vectors of entities
print ('Reading entities and obtaining vectors of entities')
model = Word2Vec.load('dbpediaVectors/dbpedia.w2v')
entityCount = len(entityToVector)
entityVecFoundCount = 0
for entity in entityToVector:
	dbEntity = 'DBPEDIA_ID/' + entity
	if(dbEntity in model.wv):
		entityToVector[entity] = model.wv[dbEntity]
		entityVecFoundCount += 1

print ('Entities without vectors: ' + str(entityCount - entityVecFoundCount))
print ('Writing the entity vectors file')
entityToVectorFile = open(entityToVectorFilePath, 'w')
for entity in entityToVector:
	currVector = entityToVector[entity]
	vectorStr = ''
	for currDim in currVector:
		vectorStr = vectorStr + ' ' + str(currDim)
	try:
		entityToVectorFile.write(entity.encode('utf-8').strip() + ' ' + vectorStr.strip() + '\n')
	except:
		continue
entityToVectorFile.close()
