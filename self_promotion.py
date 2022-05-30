from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import pandas as pd
from spacy import displacy
import nltk
from nltk.corpus import wordnet
import re
from nltk.tokenize import sent_tokenize
df = pd.read_csv('QA_Dataset.csv')
sen = df[['answer']]
temp=[]
for i in sen['answer'][1:800:2]:
    temp.extend(sent_tokenize(i))
print(temp)
# Average pos polarity score of prom true :  0.15430769230769234
# Average neg polarity score of prom true :  0.041494505494505486
# Average neu polarity score of prom true :  0.8041978021978019
sentences = [
    "I set an excellent example to the team",
    "I also worked to add an extra shuttle service to campus.",
    "I work 20 hours each week to help put myself through school",
    "I took complete charge and completely motivated the group and we were able to raise a thousand dollars.",
    "I have faced financial setbacks as tuition has increased; however, I’ve always managed to earn enough money to pay my tuition",
    "I did not manage the development team",
    "I like to code in Java",
    "I would like to be a part of this team",
    "I develop fullstack applications",
    "I led the club and organized various events",
    "I study in IIIT Bangalore",
    "I studied Operating Systems in my third year",
    "I built an image-captioning model that achieves a BLEU score of 0.85.",
    "I helped the team deliver the presentation",
    "I am extremely persuasive and I have exceptionally strong interpersonal skills."
]
print()
sentences.extend(temp)
print(sentences)
self_prom_true, self_prom_false = [], []

# Trying out if sentiment analysis can help identify the 'intensity' of self-promotion
sid = SentimentIntensityAnalyzer()
k = 0
for sentence in sentences:
    ss = sid.polarity_scores(sentence)  
    print(ss)
    if k%2:
        print("\n")
    k += 1

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
my_dict = ["create", "build", "implement", "lead", "guide", "manage", "work", "spearhead", "develop", "achieve", "improve", "mentor", "resolve", "volunteer", "influence", "launch", "win", "conceptualize", "orchestrate", "help"]
k = 0
modified_dataset = []
for sentence in sentences:
    doc = nlp(sentence)
    print(str(doc))
    pronouns = []
    verbs = [] 
    tense = [] 
    lemmatized_verbs = []
    dobj = []
    for token in doc:
        # print(token.text, token.morph)  # 'Case=Nom|Number=Sing|Person=1|PronType=Prs'

        # First person pronoun and tense checking
        if("Prs" in token.morph.get("PronType") and "1" in token.morph.get("Person")):  # ['Prs'] PRS means a personal pronoun.
            pronouns.append(token.text)
        if("Past" in token.morph.get("Tense") or "Pres" in token.morph.get("Tense")):  # ['Prs']
            tense.append(token.text)
        #cases : I am hardworking : no verb hence verb not compulsory
        #        I developed an application that was used by many people. : Adjective not compulsory
        #

    if (not pronouns): #so if there are no pronouns or tense past/pres then there cannot be self promotion.
        self_prom_false.append(sentence)
        continue
    modified_dataset.append(sentence)
    for token in doc:
        for i in pronouns:
            print(str(token))
            # print([str(child) for child in token.children])
            temp = [str(child) for child in token.children] 
            if (i in temp and 'not' not in temp):
                # For sentences like 'I helped the team develop the product', the main verb is not 'help' but it is 'develop'
                # So we should not stop at the first verb and instead look at its children
                # This list is non-exhaustive and should be extended to include all other similar verbs.


                #i is in pronouns by for condition and i is in the children of the token. then i will definitely be a pronoun below condition is wrong.
                if i not in ["helped, tried"] :
                    print("THE PART OF SPEECH IS", token.pos_)
                    if(token.pos_ == "VERB"):
                        verbs.append(token.text)
                    break
                else:
                    print("THE PART OF SPEECH IS", token.pos_)
                    temp2 = [str(child) for child in token.children] 
                    verbs.extend(temp2)
    print("THE VERBS ARE -----------") #The verbs were not getting recognised properly
    print(verbs)
            
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
        if verb in my_dict or sid.polarity_scores(sentence)['pos']>0.15430769230769234:
            self_prom_true.append(sentence)
            b = True
            break 
    if not b: self_prom_false.append(sentence)

identified_true_pospolarityscore =0
identified_true_negpolarityscore =0
identified_true_neupolarityscore =0
print("Self promoting")
for i in self_prom_true:
    print(i)
    identified_true_pospolarityscore += sid.polarity_scores(i)['pos']
    identified_true_negpolarityscore += sid.polarity_scores(i)['neg']
    identified_true_neupolarityscore += sid.polarity_scores(i)['neu']
print()
print("Average pos polarity score of prom true : ", identified_true_pospolarityscore/len(self_prom_true))
print("Average neg polarity score of prom true : ", identified_true_negpolarityscore/len(self_prom_true))
print("Average neu polarity score of prom true : ", identified_true_neupolarityscore/len(self_prom_true))
print("Not promoting")
for i in self_prom_false:
    print(sid.polarity_scores((i)))
    print(i)

print("The Dataset was modified and shortened:")
perc = len(modified_dataset)/len(sentences) *100
print("Modified data set is "+ str(perc) +"% of the size of original dataset. Sentences removed = "+str(len(sentences) - len(modified_dataset)))

#Sentiment analysis does not work well enough.