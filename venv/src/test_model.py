

import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt # type: ignore
from keras import models


# Charger le modèle entraîné
model =models.load_model("fakeNews.keras")


# Préparer les données de test
text_df = pd.read_csv("venv/src/fake_news/fake_or_real_news.csv")
text = list(text_df['text'])

# Concaténer le texte
joined_text = " ".join(text)
partial_text = joined_text[:5000]

# Tokeniser le texte
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(partial_text.lower())

# Créer le dictionnaire de tokens
unique_tokens = np.unique(tokens)
unique_tokens_index = {token: idx for idx, token in enumerate(unique_tokens)}
index_to_token = {idx: token for token, idx in unique_tokens_index.items()}

# Définir le nombre de mots utilisés pour faire une prédiction
n_words = 15

# Préparer les séquences d'entrée et de sortie pour la prédiction
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

# Encoder les séquences d'entrée et de sortie
X_test = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)
Y_test = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X_test[i, j, unique_tokens_index[word]] = 1
    Y_test[i, unique_tokens_index[next_words[i]]] = 1

# Évaluer le modèle sur les données de test
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Prédire les mots suivants en utilisant le modèle
predictions = model.predict(X_test, verbose=0)
predicted_words = [index_to_token[np.argmax(pred)] for pred in predictions]

print("predicted_words", predicted_words)
# Calculer l'accuracy
true_words = [index_to_token[np.argmax(true)] for true in Y_test]
test_accuracy = accuracy_score(true_words, predicted_words)
print(f'Test Accuracy: {test_accuracy}')

# Tracer les courbes de perte et de précision
history = model.history.history
plt.figure(figsize=(12, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Train Loss')
plt.title('Courbe de Perte')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Courbe de précision
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.title('Courbe de Précision')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# def predict_next_word(input_text,n_best):
    # input_text= 