#! /bin/python

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

input_data = [
	"I'd like an apple.",
	"An apple a day keeps the doctor away.",
	"Never compare an apple to an orange.",
	"I prefer scikit-learn to orange."
	]


# Vectorize the input data into TF-IDF vectors
vec = TfidfVectorizer()
vec = vec.fit_transform(input_data)
vec = vec.todense()

# We need to find maximum similarity between sentence 1 and 
# rest of the sentences. Use cosine similarity algorithm
# 	Sentence #1: "I'd like an apple."
#	Sentence #2: "An apple a day keeps the doctor away."
#	Sentence #3: "Never compare an apple to an orange."
#	Sentence #4: "I prefer scikit-learn to orange."
#
# Get index of max value and add to 2 it since we need to 
# print either 2 or 3 or 4 instead of 0, 1, 2.
ans = cosine_similarity(vec[0], vec[1:])
print np.argmax(ans) + 2