import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = { 
        'Movie_ID': [1, 2, 3, 4, 5],
        'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
        'Genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama']}

#convert dataset into dataframe
df = pd.DataFrame(data)

#Display dataset
print("Movie Dataset:")
print(df)

#Define a TF-IDF Vectorizer to convert the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

#Fit and transform the genre column into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['Genre'])

#Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Function to get movie recommendations based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    #Get the index of the movie that matches the title
    idx = df.index[df['Title'] == title][0]
    
    #Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Get the scores of the 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]
    
    #Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    #Return the top 3 most similar movies
    return df['Title'].iloc[movie_indices]

#Test the recommendation system with an example
movie_title = 'The Godfather'
recommended_movies = get_recommendations(movie_title)
print(f"\nMovies recommended for '{movie_title}':")
for movie in recommended_movies:
    print(movie)