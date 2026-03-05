import numpy as np
import pandas as pd

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('/content/anime.csv')
df.head()
# Fill missing values

df['title'] = df['title'].fillna('')
df['synopsis'] = df['synopsis'].fillna('')

# Combine features

df['combined'] = df['title'] + " " + df['synopsis']

df.head()
vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = vectorizer.fit_transform(df['combined'])

print(tfidf_matrix.shape)
def search_anime(anime_name):

    # convert input to vector
    input_vector = vectorizer.transform([anime_name])

    # calculate similarity
    similarity = cosine_similarity(input_vector, tfidf_matrix)

    # get index of best match
    index = similarity.argmax()

    # get anime details
    anime = df.iloc[index]

    print("Anime Found:\n")

    print("Title:", anime['title'])

    if 'score' in df.columns:
        print("Score:", anime['score'])

    if 'rank' in df.columns:
        print("Rank:", anime['rank'])

    if 'episodes' in df.columns:
        print("Episodes:", anime['episodes'])

    if 'popularity' in df.columns:
        print("popularity:", anime['popularity'])

    print("\nSynopsis:\n")

    print(anime['synopsis'])
