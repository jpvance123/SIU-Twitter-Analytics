#!/usr/bin/env python3
import numpy as np
import pandas as pd
import re
import warnings

#Visualisation
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from IPython.display import display

from wordcloud import WordCloud, STOPWORDS

#nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize

matplotlib.style.use('ggplot')
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")



train = pd.read_csv('tweets.csv',names=["id","keyword","location","Tweet","target"])
print(train)
train.info()
train.isna().groupby('Tweet').count()


for i in range(len(train['Tweet'])):
	train['Tweet'][i] = " ".join([word for word in train['Tweet'][i].split() if 'http' not in word and '@' not in word and '<' not in word])

train['Tweet'] = train['Tweet'].apply(lambda x: re.sub('[!@#$->,;:?]', '', x.lower())) 
 #   except AttributeError:    
#        tweets['tweetos'][i] = 'other'

#Preprocessing tweetos. select tweetos contains 'RT @'
#for i in range(len(tweets['Tweet'])):
   # if tweets['tweetos'].str.contains('@')[i]  == False:
        #tweets['tweetos'][i] = 'other'
        
# remove URLs, RTs, and twitter handles
#for i in range(len(tweets['Tweet'])):
  #  tweets['Tweet'][i] = " ".join([word for word in tweets['Tweet'][i].split()
                          #      if 'http' not in word and '@' not in word and '<' not in word])


#tweets['Tweet'][1]

#tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
#tweets['Tweet'] = tweets['Tweet'].apply(lambda x: re.sub('  ', ' ', x))
#tweets['Tweet'][1]

stopwords = set(STOPWORDS)
tweet = " ".join(word for word in train['Tweet'])
wordcloud = WordCloud(background_color="white",stopwords=stopwords).generate(tweet)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Disaster Analysis")
plt.show()
#wordcloud(tweets,'Tweet') 

import random
train['random_number'] = np.random.randn(len(train.index))

new_train = train[train['random_number'] <= 0.8]
new_test = train[train['random_number'] >0.8]

print(new_train.shape)
print(new_test.shape)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern = r'\b\w+\b')

new_train_matrix = vectorizer.fit_transform(new_train['Tweet'])
new_test_matrix = vectorizer.transform(new_test['Tweet'])

x_train = new_train_matrix
x_test = new_test_matrix
y_train = new_train['target']
y_test = new_test['target']

#from sklearn.neighbors import kNeighborsClassifier

#model = KNeighborsClassifier(n_neighbors = 7)
#model.fit(x_train, y_train)

#predictions2 = model.predict(x_test)

#from sklearn.metrics import confusion_matrix, classification_report

#print(confusion_matrix(predictions2, y_test))
#print('*'*40)
#print(classification_report(prediction2, y_test))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

predictions3 = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(predictions3, y_test))
print('*'*40)
print(classification_report(predictions3, y_test))