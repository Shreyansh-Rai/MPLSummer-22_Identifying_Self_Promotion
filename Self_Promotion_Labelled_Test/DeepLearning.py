import pandas as pd
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

# Read in data
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
print(tokenizer.word_index)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# Create an LSTM model with an Embedding layer and fit training data
model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,\
                           output_dim=100,\
                           input_length=60))  #Turns the positive numbers into dense vectors
model.add(layers.Bidirectional(layers.LSTM(128)))
model.add(layers.Dense(2,activation='softmax'))
model.compile(optimizer='adam',\
              loss='categorical_crossentropy',\
              metrics=['accuracy'])
model.fit(X_train,\
          y_train,\
          batch_size=128,\
          epochs=50,\
          validation_data=(X_test,y_test))