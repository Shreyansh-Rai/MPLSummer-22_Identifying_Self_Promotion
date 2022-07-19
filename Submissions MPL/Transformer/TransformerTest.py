import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
# uploaded = files.upload()
# Read in data io.BytesIO(uploaded['400sentences.csv'])
data=pd.read_csv('400sentences.csv',sep=";")

# Create all the features to the data set
def data_cleaning(text_list):
    stopwords_rem=False
    stopwords_en=stopwords.words('english')
    lemmatizer=WordNetLemmatizer()
    tokenizer=TweetTokenizer()
    reconstructed_list=[]
    for each_text in text_list:
        lemmatized_tokens=[]
        tokens=tokenizer.tokenize(each_text.lower())
        pos_tags=pos_tag(tokens)
        for each_token, tag in pos_tags:
            if tag.startswith('NN'):
                pos='n'
            elif tag.startswith('VB'):
                pos='v'
            else:
                pos='a'
            lemmatized_token=lemmatizer.lemmatize(each_token, pos)
            if stopwords_rem: # False
                if lemmatized_token not in stopwords_en:
                    lemmatized_tokens.append(lemmatized_token)
            else:
                lemmatized_tokens.append(lemmatized_token)
        reconstructed_list.append(' '.join(lemmatized_tokens))
    return reconstructed_list


# Break data down into a training set and a testing set
X = data['ans']
y = data['issp']
print(len(X))
X_train, X_test, y_train, y_test=train_test_split(X, y,stratify=y, test_size=0.05, random_state=10)
print("length of the train size: ", len(X_train))
# Fit and transform the data
X_train=data_cleaning(X_train)
X_test=data_cleaning(X_test)
print(len(X_train) + len(X_test))
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size=len(tokenizer.word_index)+1
print(f'Vocab Size: {vocab_size}')
#We will take the sentences and then find out the vocab size ie the more freq words get smaller index
#something similar to creating a dictionary of words where the more freq words appear first.
X_train=pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=60) #padding the sentences to make them of equal len
X_test=pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=60)#also maps words to sequence in the X_train and test sequences.
# print(tokenizer.word_index)
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

# Create an LSTM model with an Embedding layer and fit training data
# model=Sequential()
# model.add(layers.Embedding(input_dim=vocab_size,\
#                            output_dim=100,\
#                            input_length=60))  #Turns the positive numbers into dense vectors
# model.add(layers.Bidirectional(layers.LSTM(128)))
# model.add(layers.Dense(2,activation='softmax'))
# model.compile(optimizer='adam',\
#               loss='categorical_crossentropy',\
#               metrics=['accuracy'])
# model.fit(X_train,\
#           y_train,\
#           batch_size=128,\
#           epochs=40,\
#           validation_data=(X_test,y_test))

maxlen = 60

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
  
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train, y_train, batch_size=128, epochs=150, validation_data=(X_test, y_test)
)
