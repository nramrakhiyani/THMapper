# THMapper
# Major Project
# CSE474 - Information Retrieval and Extraction

Kindly refer to the Repo Wiki for details like problem description, team details and approches developed.

README
This repo consists of the various code and data developed as part of the project.

map.py: Creates the dataset of tail to head query mapping based on a heuristic.
preparePairs.py and parse.py: Prepares the tail query head query mapping file to make it ready for consumption by neural network code
getEntityDBPediaVectors.py: Obtains the the DBPedia embeddings for entities.

Approach 1: Entity Linking
mydexter.py: Computes the score for each head query given a tail query using the entity linking method based on Dexter

Approach 2:
THMapNet_v1.py: First Neural Network architecture used for learning the mappings
THMapNet_v2.py: Second Neural Network architecture used for learning the mappings
THMapNet_Server_Test.py: Computes the score for a set of head queries given a tail query using THMapNet_v1
