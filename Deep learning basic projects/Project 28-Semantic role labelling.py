# Install if not already:
# pip install allennlp allennlp-models
 
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
 
# Load pretrained SRL model from AllenNLP
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
 
# Input sentence
sentence = "John gave a book to Mary on her birthday."
 
# Perform Semantic Role Labeling
result = predictor.predict(sentence=sentence)
 
# Display predicates and their arguments
print("\nSentence:", sentence)
print("\nSemantic Roles:\n")
for verb in result['verbs']:
    print(f"▶ Verb: {verb['verb']}")
    print(f"   Tags: {' '.join(verb['tags'])}")