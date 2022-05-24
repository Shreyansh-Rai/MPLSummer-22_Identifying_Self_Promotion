import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import re
data = pd.read_csv('google_play_store_apps_reviews_training.csv')
print(data.head())

def cleanup(data):
    data = data.drop('package_name',axis = 1)
    return data


def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('package_name', axis=1)

    # Convert text to lowercase
    data['review'] = data['review'].str.strip().str.lower()
    return data

def strclean(s):
    removelist="" #additional things to remove if needed.
    result = re.sub('<.*?>','',s);
    result = re.sub('https://.*', '', result)
    result = re.sub(r'[^a-zA-Z0-9]', ' ', result)  # removing non alpha numeric characters.
    result = result.lower()
    return result

data = preprocess_data(data)
data['review'] = data['review'].apply(strclean)
print(data)
x = data['review']
y = data['polarity']

x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)
#to maintain a similar proportion of positive and negative reviews in the train and test set.
# https://www.educative.io/edpresso/countvectorizer-in-python
# Vectorize text reviews to numbers
# The vectorise creates a sparse matrix as shown below.
#             word1 word2 word3 word4 ... wordn
# doc0 frq     .     .     .     .    ...   .
# doc1 frq     .     .     .     .    ...   .
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x, y)
print(model.score(x_test,y_test))
print(model.predict(vec.transform(['fun app'])))