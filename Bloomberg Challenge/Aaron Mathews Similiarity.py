# Team Members: Aaron Mathews and Clayton Dumay

# Importing everything
from scipy import spatial
import pandas as pd

import json

file = open('word_embeddings.json', 'r')
word_embeddings_list = json.load(file)


def similarity_measure(x, y):
    # Squared euclidean gives higher weight to closer values and multiplyting with cosine provides a more diverse classificiation
    return spatial.distance.euclidean(x, y)


# First step is to read in the csv data for both CNN and the Federal Samples. This will be stored in 2 pandas dataframes
cnnDf = pd.read_csv('cnn_samples-54b19b96f3c0775b116bad527df8c7b5.csv')
cnnDf.columns = ['id', 'source', 'text', 'embeddings']
fedDf = pd.read_csv('federal_samples-a586d0681e005629453435bea5b173eb.csv')
combinedDf = pd.concat([cnnDf, fedDf], ignore_index=True)


# Next, read in the challenge data points and store in a separate dataframe.
challengeDf = pd.read_csv('challenge-ddec63cf66ea88f128e3c21e457f393a.csv')

# This list will store tuples for each row of input data. The first element will be the distance, the second element will be the row index, and the third element will be a string representing the dataset
similaritiesList = []
# The number of neighbors that will be considered for the k nearest neighbors algorithm
k = 9

# This list will store the k most similar texts for each challenge embedding
mostSimilarTexts = []
for i in range(5):
    mostSimilarTexts.append([])
    for j in range(k):
        mostSimilarTexts[i].append([])

# A list of lists storing k closest words for each challenge embedding
keywords = []
for i in range(5):
    keywords.append([])
    for j in range(k):
        keywords[i].append([])
challengeNumber = 0
# Initially, all the embeddings are stored as strings in the dataframe
for challengeEmbeddingString in challengeDf['embeddings']:
    index = 0
    # Converting the embedding string to a list of floats
    challengeEmbeddingStringList = challengeEmbeddingString[1:len(challengeEmbeddingString)-1].split(',')
    challengeEmbedding = [float(i) for i in challengeEmbeddingStringList]
    # Looping through all the embeddings in the cnn and fed samples dataset and calculating and storing similarities
    for EmbeddingString in combinedDf['embeddings']:
        EmbeddingStringList = EmbeddingString[1:len(EmbeddingString)-1].split(',')
        Embedding = [float(i) for i in EmbeddingStringList]
        currentSimilarityMeasure = similarity_measure(challengeEmbedding, Embedding)
        # Adding the current similarity measure and identifying data to the similarities list
        similaritiesList.append((currentSimilarityMeasure, index))
        index += 1
    index = 0
    # Sorting the similarities based on similarity distance
    similaritiesList.sort()
    # Storing the k closest text data points
    for i in range(k):
        mostSimilarTexts[challengeNumber][i] = combinedDf['text'][similaritiesList[i][1]]
    similaritiesList.clear()
    keywordSimilarities =[]
    for keywordEmbedding in word_embeddings_list:
        currentSimilarityMeasure = similarity_measure(keywordEmbedding[0], challengeEmbedding)
        keywordSimilarities.append((currentSimilarityMeasure, keywordEmbedding[1]))
    keywordSimilarities.sort()
    for i in range(k):
        keywords[challengeNumber][i] = keywordSimilarities[i][1]
    keywordSimilarities.clear()
    challengeNumber += 1

for i in range(5):
    print(f'Challenge {i} Most Similar Text Matchings')
    for j in range(k):
        print(mostSimilarTexts[i][j])
        print('')
    for j in range(k):
        print(keywords[i][j])


# # Mystery Challenge
# mystery = json.loads()