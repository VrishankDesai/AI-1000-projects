import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

#Download NLTK data files
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

#Preprocess the data and extract features
def extract_features(words):
    return {word: True for word in words}

#Load movie_reviews in dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

#Shuffle the dataset to ensure random distribution
random.shuffle(documents)

#Prepare dataset for training and testing
featuresets = [(extract_features(doc), category) for (doc, category) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]

#Train the Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

#Evaluate the classifier on the test set
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")
      
#Show the most informative features
classifier.show_most_informative_features(10)

#Test on new input sentences
def analyze_sentiment(text):
    #Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]  # Remove punctuation
    
    #Predict sentiment
    features = extract_features(words)
    return classifier.classify(features)
    
#Test the classifier with custom text inputs
test_sentences = [
    "I absolutely loved this movie! The plot was thrilling and the characters were well-developed.",
    "This was a terrible film. I wasted two hours of my life.",
    "An average movie with some good moments but overall forgettable.",
    "Fantastic performance by the lead actor, truly captivating!",
    "The storyline was dull and predictable."
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}")
    print()