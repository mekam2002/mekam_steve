
import pandas as pd

import numpy as np
from nltk.tokenize import RegexpTokenizer

from keras import models
from keras import layers
from keras import optimizers

text_df = pd.read_csv("venv/src/fake_news/fake_or_real_news.csv")
text = list(text_df['text'])

# this will delete all blank spaces between characters
joined_text = " ".join(text)

partial_text = joined_text[:10000]
# This is the regular expression pattern passed to the RegexpTokenizer. It is a raw string
# meaning that Python will not treat backslashes as escape characters.
# r'...': The r indicates a raw string in Python. This ensures that backslashes
# are treated literally and not as escape characters.
# \w: This matches any word character, which includes letters 
# (uppercase and lowercase), digits, and underscores.
# +: This quantifier means "one or more" of the preceding element. 
# In this case, it means "one or more word characters."
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(partial_text.lower( ))

# we count all the unique tokens in the text or the unique word
unique_tokens = np.unique(tokens)

# ater that we create an universal dictionary where we save each unique token
#  and it positon in the raw text
unique_tokens_index = {token:idx for idx ,token in enumerate(unique_tokens)}


# the number of words we will look back to make the prediction
n_words = 15
input_words = []

# the selection range of the next words
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i+n_words])
    next_words.append(tokens[i+n_words])
    
X = np.zeros((len(input_words),n_words,len(unique_tokens)),dtype=bool)
Y = np.zeros((len(next_words),len(unique_tokens)),dtype=bool)

for i ,words in enumerate(input_words):
    for j , word in enumerate(words):
        X[i,j,unique_tokens_index[word]] = 1
    Y[i,unique_tokens_index[next_words[i]]] = 1
    
model = models.Sequential()
model.add(layers.LSTM(128,input_shape=(n_words,len(unique_tokens)),return_sequences=True))
model.add(layers.LSTM(128))
model.add(layers.Dense(len(unique_tokens)))
model.add(layers.Activation('softmax'))


model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=1e-3),
    metrics=['accuracy']
)
model.fit(X, Y,batch_size=128 ,epochs =30,shuffle=True)

model.save("fakeNews.keras")