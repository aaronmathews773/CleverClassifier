from assets.datathon_api import get_embedding
# Using the english-words library: https://pypi.org/project/english-words/
from english_words import english_words_set
import time
import json

# Generating possible keywords using common words
api_key = "hlNYN7PiSp9vdc5RTJ0iNEoXWrpz8WV1qUW0wtk8"

word_list = list(english_words_set)

file = open('word_embeddings.json', 'a')

word_embeddings_list = []

for i in range(120):
    word_embeddings_list.append((get_embedding(word_list[i], api_key), word_list[i]))
    time.sleep(1.5)

json.dump(word_embeddings_list, file)