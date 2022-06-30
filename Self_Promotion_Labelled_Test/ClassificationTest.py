import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
# data = pd.read_csv('100RandomSentencesCurrent.csv',sep=";")
data = pd.read_csv('400Sentences.csv',sep=";")
print(data.head())

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


def cleanup(data):
    data = data.drop('issp_type',axis = 1)
    return data


def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('issp_type', axis=1)

    # Convert text to lowercase
    data['ans'] = data['ans'].str.strip().str.lower()
    return data

def strclean(s):
    removelist="" #additional things to remove if needed.
    result = re.sub('<.*?>','',s);
    result = re.sub('https://.*', '', result)
    result = re.sub(r'[^a-zA-Z0-9]', ' ', result)  # removing non alpha numeric characters.
    result = result.lower()
    return result

data = preprocess_data(data)
data['ans'] = data['ans'].apply(strclean)
print(data)
x = data['ans']
y = data['issp']

x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.05, random_state=10)
# x=data_cleaning(x)
# x_test=data_cleaning(x_test)
#to maintain a similar proportion of positive and negative reviews in the train and test set.
# https://www.educative.io/edpresso/countvectorizer-in-python
# Vectorize text reviews to numbers
# The vectorise creates a sparse matrix as shown below.
#             word1 word2 word3 word4 ... wordn
# doc0 frq     .     .     .     .    ...   .
# doc1 frq     .     .     .     .    ...   .

vec = CountVectorizer(stop_words='english',ngram_range=(1,3))

# vec = TfidfVectorizer(max_features=1000000, ngram_range=(1, 10))
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()
from sklearn.naive_bayes import MultinomialNB,GaussianNB,ComplementNB,CategoricalNB,BernoulliNB

# model = MultinomialNB(alpha=4.1,fit_prior=True,class_prior=None)
model = MultinomialNB()
# model2= GaussianNB()
# model3=ComplementNB()
# model4=CategoricalNB()
# model5=BernoulliNB()
model.fit(x, y)
print(model.score(x_test,y_test))
print(model.predict(vec.transform(['I started working very hard and got where I am today'])))
print(f'Test Score: {model.score(x_test, y_test)}')
print(f'Train Score: {model.score(x, y)}')

#Word to vec try pls