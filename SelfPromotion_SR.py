from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import pandas as pd
from spacy import displacy
import nltk
from nltk.corpus import wordnet
import re
from nltk.tokenize import sent_tokenize

df = pd.read_csv('100RandomSentencesCurrent.csv',sep=";")
df = df.drop('issp_type',axis=1)
sen = df['ans']
sen[4] = "I established the firm"
pol = df['issp']
print(df.head())
sentences = []
sentences.extend(sen)
print(sentences)
self_prom_true, self_prom_false = [], []

# Trying out if sentiment analysis can help identify the 'intensity' of self-promotion
sid = SentimentIntensityAnalyzer()
k = 0
# for sentence in sentences:
#     ss = sid.polarity_scores(sentence)
#     print(ss)
#     if k % 2:
#         print("\n")
#     k += 1
polsp =[]
polnsp = []
statssp=0
statsnsp = 0
nlp = spacy.load('en_core_web_sm')

for sentence in sentences:
    text = nlp(sentence)
    # for token in text:
    #     print(token.text, '=>',token.pos_,'=>',token.tag_)
    # print()

    # Uncomment this to render the POS tree
    # displacy.render(text, jupyter=True)
    # displacy.serve(text, style="dep")

# Defining a list of verbs that could potentially be relevant from an interview standpoint
def getmy_dict(my_dict) :
    my_dict = ["create", "build", "implement", "lead", "guide", "manage", "work", "spearhead", "develop", "achieve",
               "improve", "mentor", "resolve", "volunteer", "influence", "launch", "win", "conceptualize",
               "orchestrate",
               "help"]
    print("Data from positive verbs csv")

    md = []
    md = pd.read_csv("PositiveActionWordscsv.csv")
    md = md['List']
    regex = re.compile('[^a-zA-Z]')
    for i in md:
        my_dict.append(regex.sub('', i))
    return my_dict

my_dict = []
lemdic=[]
my_dict = getmy_dict(my_dict)
for i in range(len(my_dict)):
    verb = nlp(my_dict[i])
    lemdic.extend([token.lemma_ for token in verb])
my_dict = lemdic
print(my_dict)
k = 0
modified_dataset = []
counter = -1
for sentence in sentences:
    counter += 1
    print(sentence, " ", pol[counter])
    doc = nlp(sentence)
    # print(str(doc))
    pronouns = []
    verbs = []
    tense = []
    lemmatized_verbs = []
    dobj = []
    for token in doc:
        # print(token.text, token.morph)  # 'Case=Nom|Number=Sing|Person=1|PronType=Prs'
        # First person pronoun and tense checking
        if ("Prs" in token.morph.get("PronType") and "1" in token.morph.get("Person")):  # ['Prs'] PRS means a personal pronoun.
            pronouns.append(token.text)
        if ("Past" in token.morph.get("Tense") or "Pres" in token.morph.get("Tense")):  # ['Prs']
            tense.append(token.text)
        # if (str(token) in ["will","would", "might", "could", "should", "shall"]) : #there is no future tense tag in token
        #     self_prom_false.append(sentence)
        #     print("False")
        #     polnsp.append(pol[counter])
        #     if (pol[counter] == 0):
        #         statsnsp += 1
        #     continue

    if (not pronouns):  # so if there are no pronouns or tense past/pres then there cannot be self promotion.
        self_prom_false.append(sentence)
        print("False")
        polnsp.append(pol[counter])
        if (pol[counter] == 0):
            statsnsp += 1
        continue
    modified_dataset.append(sentence)
    for token in doc:
        for i in pronouns:
            #print(str(token))
            # print([str(child) for child in token.children])
            temp = [str(child) for child in token.children]
            if (i in temp and 'not' not in temp):
                # For sentences like 'I helped the team develop the product', the main verb is not 'help' but it is 'develop'
                # So we should not stop at the first verb and instead look at its children
                # This list is non-exhaustive and should be extended to include all other similar verbs.

                # i is in pronouns by for condition and i is in the children of the token. then i will definitely be a pronoun below condition is wrong.
                if i not in ["helped, tried"]:
                    # print("THE PART OF SPEECH IS", token.pos_)
                    if (token.pos_ == "VERB"):
                        verbs.append(token.text)
                    break
                else:
                    # print("THE PART OF SPEECH IS", token.pos_)
                    temp2 = [str(child) for child in token.children]
                    verbs.extend(temp2)
    # print("THE VERBS ARE -----------")  # The verbs were not getting recognised properly
    # print(verbs)

    # for token in doc:
    # print(token, token.morph)
    # print(verbs)

    # Lemmatizing each verb (finding its root word) for easy checking
    for i in range(len(verbs)):
        verb = nlp(verbs[i])
        lemmatized_verbs.extend([token.lemma_ for token in verb])

    # Synonyms
    synonyms = set()
    for word in my_dict:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.add(l.name())

    b = False
    for verb in lemmatized_verbs:
        if verb in my_dict:   #or sid.polarity_scores(sentence)['pos'] > 0.15430769230769234:
            self_prom_true.append(sentence)
            print("True")
            polsp.append(pol[counter])
            if(pol[counter] == 1) :
                statssp +=1
            b = True
            break
    if not b:
        self_prom_false.append(sentence)
        polnsp.append(pol[counter])
        print("False")
        if(pol[counter] == 0):
            statsnsp +=1


identified_true_pospolarityscore = 0
identified_true_negpolarityscore = 0
identified_true_neupolarityscore = 0
print("Self promoting")
for i in self_prom_true:
    print(i)
    identified_true_pospolarityscore += sid.polarity_scores(i)['pos']
    identified_true_negpolarityscore += sid.polarity_scores(i)['neg']
    identified_true_neupolarityscore += sid.polarity_scores(i)['neu']
print()
print("Average pos polarity score of prom true : ", identified_true_pospolarityscore / len(self_prom_true))
print("Average neg polarity score of prom true : ", identified_true_negpolarityscore / len(self_prom_true))
print("Average neu polarity score of prom true : ", identified_true_neupolarityscore / len(self_prom_true))
print("Not promoting")
for i in self_prom_false:
    print(sid.polarity_scores((i)))
    print(i)

print("The Dataset was modified and shortened:")
perc = len(modified_dataset) / len(sentences) * 100
print("Modified data set is " + str(perc) + "% of the size of original dataset. Sentences removed = " + str(
    len(sentences) - len(modified_dataset)))

print("Size of self promotion true: ", len(self_prom_true), "correct hits = ",statssp, "Actual : 31 self promotional")
print("Size of self promotion false: ", len(self_prom_false), "correct hits = ",statsnsp)
# Sentiment analysis does not work well enough.