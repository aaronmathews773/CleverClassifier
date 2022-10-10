from scipy import spatial
import pandas as pd
from assets.datathon_api import get_embedding
import time
# Uses the food reviews dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

api_key = "hlNYN7PiSp9vdc5RTJ0iNEoXWrpz8WV1qUW0wtk8"


def similarity_measure(x, y):
    # Squared euclidean gives higher weight to closer values and multiplyting with cosine provides a more diverse classificiation
    return spatial.distance.cosine(x, y)

food_reviews = pd.read_csv('Reviews.csv')

# A list of the potential keywords that a review could be
keyword_list = ['amazing', 'good', 'decent', 'alright', 'bad', 'awful', 'panda']
keyword_embeddings = []

# Creating embeddings for all the keywords
for i in range(len(keyword_list)):
    keyword_embeddings.append(get_embedding(keyword_list[i], api_key))
    time.sleep(1)

# Number of keywords generated for each review
k = 2

# Number of reviews being evaluated
n = 5

# List of lists storing k keywords for each of the n review texts
review_keywords = []
for i in range(n):
    review_keywords.append([])

review_number = 0
for review_text in food_reviews['Text'].head(n):
    review_similarities = []
    review_embedding = get_embedding(review_text, api_key)
    time.sleep(1)
    for i in range(len(keyword_embeddings)):
        review_similarities.append((similarity_measure(review_embedding, keyword_embeddings[i]), keyword_list[i]))
    review_similarities.sort()
    # Storing the k nearest keywords for each review
    for i in range(k):
        review_keywords[review_number].append(review_similarities[i][1])
    review_number += 1
    review_similarities.clear()

# Printing the top keywords for each food review
for i in range(n):
    print(f'Keywords for Review {i}')
    print('Review Text:', food_reviews['Text'][i])
    for j in range(k):
        print(review_keywords[i])






