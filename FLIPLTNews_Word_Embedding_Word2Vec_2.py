#!/usr/bin/env python
# coding: utf-8

#                      NLP: FlipItNews By Ratnesh Kumar 
#                      

# Context:
# 
# The Gurugram-based company ‘FlipItNews’ aims to revolutionize the way Indians perceive finance, business, and capital market investment, by giving it a boost through artificial intelligence (AI) and machine learning (ML). They’re on a mission to reinvent financial literacy for Indians, where financial awareness is driven by smart information discovery and engagement with peers. Through their smart content discovery and contextual engagement, the company is simplifying business, finance, and investment for millennials and first-time investors

# **Problem Statement:**
# 
#   The goal of this project is to use a bunch of news articles extracted from the companies’ internal database and categorize them into several categories like politics, technology, sports, business and entertainment based on their content. Use natural language processing and create & compare at least three different models.

# Attribute Information:
# 
# Article
# 
# Category
# 
# The features names are themselves pretty self-explanatory
# 
# Concepts Tested:
# 
# Natural Language Processing
# 
# Text Processing
# 
# Stopwords, Tokenization, Lemmatization
# 
# Bag of Words, TF-IDF
# 
# Multi-class Classification

# Evaluation Criteria (100 points)
# 
# 1. Importing the libraries & Reading the data file (10 points)
# 
# 2. Exploring the dataset (10 points)
# 
# Shape of the dataset
# 
# News articles per category
# 
# 3. Processing the Textual Data i.e. the news articles (30 points)
# 
# Removing the non-letters
# 
# Tokenizing the text
# 
# Removing stopwords
# 
# Lemmatization
# 
# 4. Encoding and Transforming the data (20 points)
# 
# Encoding the target variable
# 
# Bag of Words
# 
# TF-IDF
# 
# Train-Test Split
# 
# 5. Model Training & Evaluation (30 points)
# 
# Simple Approach
# 
# Naive Bayes
# 
# Functionalized Code (Optional)
# 
# Decision Tree
# 
# Nearest Neighbors
# 
# Random Forest

# #Evaluation Criteria (100 points)
# 
#     

#     1. Importing the libraries & Reading the data file (10 points)
#       

# In[2]:


import pandas as pd
df=pd.read_csv('flipitnews-data.csv')
#df=df.head(30)
df.head()


#     2. Exploring the dataset (10 points)
#           *   Shape of the dataset
#           *   News articles per category

# In[2]:


print('Shape of Dataset - ',df.shape)
print('Total No. of News Articles - ',df['Category'].nunique(),':',df['Category'].unique())
print('News Article per category - \n',df['Category'].value_counts())
print('Shape of Dataset - ',df.shape)
#print('\nInfo - \n',df.info)


# **Categorical to Numerical Encoding**
# Now, we will map each of these categories to a number, so that our model can understand it in a better way and we will save this in a new column named ‘category_id’. Where each of the categories are represented in numerical.
# 

# In[59]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Category_id']=le.fit_transform(df['Category'])


# **Data Visualisation**

# In[4]:


import matplotlib.pyplot as plt
plt.figure(figsize=(4,2))
df['Category'].hist()


#       3. Processing the Textual Data i.e. the news articles (30 points)
# 
#           Removing the non-letters
# 
#           Tokenizing the text
# 
#           Removing stopwords
# 
#           Lemmatization

# In[60]:


#*************************************Removing Stopwords *******************************
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#print(stopwords.words('english'))
def remove_stopwords(text):
  clean_text=' '.join([i for i in text.split() if i not in stopwords.words('english')])
  return clean_text

df['Article']=df['Article'].apply(lambda x:remove_stopwords(x))

#*************************************Removing Punctuations *******************************
import string

def remove_punctuation(text):
  cleantext=''.join([i for i in text if i not in string.punctuation])
  return cleantext

df['Article']=df['Article'].apply(lambda x:remove_punctuation(x))

#*************************************Lowering the Text *******************************
df['Article']=df['Article'].str.lower()

#*************************************Stemming *******************************
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stemming(text):
  clean_text=' '.join([ps.stem(i) for i in text.split()])
  return clean_text

df['Article']=df['Article'].apply(lambda x:stemming(x))

#*************************************Lemmatization *******************************
from nltk.stem import WordNetLemmatizer
wl=WordNetLemmatizer()
nltk.download('wordnet')
def Lemmatize(text):
  clean_text=' '.join([wl.lemmatize(i) for i in text.split()])
  return clean_text

df['Article']=df['Article'].apply(lambda x:Lemmatize(x))
df['Article']


# In[3]:


#*************************************Tokenization *******************************

import nltk
from nltk.tokenize import word_tokenize ,sent_tokenize
nltk.download('punkt')

word_cnt,unique_word_cnt=0,0

#Corpus of the entire Document
corpus=df['Article'].str.cat(sep=', ')
#print('corpus:',corpus)
for i in corpus:
  word_cnt+=1
print('Number of words in the entire corpus:',word_cnt)
#Find the letters used in Corpus
Unique_char=set(df['Article'].str.cat(sep=', '))
#print('Unique letters used in corpus:',Unique_char)

Vocabulary = df['Article'].str.cat(sep=', ')
print(set(word_tokenize(Vocabulary)))
for i in set(word_tokenize(Vocabulary)):
  unique_word_cnt+=1
print('Number of words in the vocabulary:',unique_word_cnt)


# **4. Encoding and Transforming the data (20 points)**
# 
#     Encoding the target variable
# 
#     Bag of Words
# 
#     TF-IDF
# 
#     Train-Test Split

# In[13]:


#*************************************One hot Encoding *******************************
def get_one_hot_vectors():
  import numpy as np

  samples = df['Article']
  # Create an empty dictionary
  token_index = {}
  #Create a counter for counting the number of key-value pairs in the token_length
  counter = 0

  # Select the elements of the samples which are the two sentences
  for sample in samples:
    for considered_word in sample.split():
      if considered_word not in token_index:

        # If the considered word is not present in the dictionary token_index, add it to the token_index
        # The index of the word in the dictionary begins from 1
        token_index.update({considered_word : counter + 1})

        # updating the value of counter
        counter = counter + 1
  print(token_index)
  # Set max_length to 6
  max_length =max(samples.str.len())
  # Create a tensor of dimension 3 named results whose every elements are initialized to 0
  results  = np.zeros(shape = (len(samples),max_length,max(token_index.values())))
  # Now create a one-hot vector corresponding to the word
  # iterate over enumerate(samples) enumerate object
  for i, sample in enumerate(samples):
    #print(i,sample)
  # Convert enumerate object to list and iterate over resultant list
    for j, considered_word in list(enumerate(sample.split())):
      #print(j,considered_word)

      # set the value of index variable equal to the value of considered_word in token_index
      index = token_index.get(considered_word)-1
      #print('index',index)
      # In the previous zero tensor: results, set the value of elements with their positional index as [i, j, index] = 1.
      results[i, j, index] = 1.
  #for j, considered_word in list(enumerate(sample.split())):
  # print(j, considered_word)
  print(samples)
  print(results[0])


# In[8]:


#*************************************Bag of Words *******************************
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
bow=cv.fit_transform(df['Article'])
print("\nVocabulary:")
print(cv.vocabulary_)

# Display the BOW matrix and vocabulary
print("Bag of Words Matrix:",bow.toarray().shape)
print(bow.toarray())
#print('Frequency of words',bow.toarray().sum(axis=0))
#print('words in vocabulary',cv.get_feature_names_out())


# In[9]:


#*************************************TFIDF - Term Frequency/Inverse Document Frequency *******************************
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
#print(tfidf.idf_)
TFIDF=tfidf.fit_transform(df['Article'])
print("\nVocabulary:")
print(tfidf.vocabulary_)

# Display the BOW matrix and vocabulary
print("TFIDF Matrix:",TFIDF.toarray().shape)
print(TFIDF.toarray())


# In[10]:


#*************************************Train-test split *******************************
X=df['Article']
y=df['Category_id']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Enable the line no. with Ctrl+shift+M
# 
#     Line 7: Iterate the corpus
#     Line 9: Set the ith word as the target word
#     Line 14,21,27 : Condition to check if the ith word in line 9 is the (first :Best) , (middle : way) or the (last : persistence) word .
#     Line 17 : If it is the first word, get the next 2 (window_size =2) words and set them as context words
#     Line 21 : If it is the last word, get the previous 2 (window_size =2) words and set them as context words
#     Line 30,37 : If our ith word is a middle word, then we need to get 2 (window_size =2) words before the ith word and 2 (window_size =2 ) words after the ith word and set all 4 as the context words. If there is only 1 word before or after the ith word, we get only 1 word.

# In[10]:


def conf_matrix(y_test, y_pred):
  from sklearn.metrics import confusion_matrix
  import seaborn as sns
  conf_mat = confusion_matrix(y_test, y_pred)

  sns.heatmap(conf_mat, annot=True, xticklabels=df['Category'].unique(), yticklabels=df['Category'].unique(), cmap="YlGnBu", fmt='g')


# #Exploring Multi-classification Models(classification models)

# **5. Model Training & Evaluation (30 points)**
# 
#     Simple Approach - Naive Bayes
# 
#     Functionalized Code (Optional)- Decision Tree,Nearest Neighbors,RandomForest

# **1. Naive Bayes - MultinomialNB**

# In[63]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X = df['Article']
y = df['Category']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

#%%time
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
conf_matrix(y_test, y_pred)


#   **2. SGDClassifier**
# 
#     Linear Support Vector Machine is widely regarded as one of the best text
#     classification algorithms.

# In[64]:


from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)

#%%time

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
conf_matrix(y_test, y_pred)


# **Logistic Regression**
#     
#     Simple and easy to understand classification algorithm, and Logistic regression can be easily generalized to multiple classes.

# In[76]:


from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)),
               ])
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
conf_matrix(y_test, y_pred)


# In[118]:


complaint = "games maker fights for survival one of britain s largest independent game makers  argonaut games  has been put up for sale.  the london-based company behind the harry potter games has sacked about 100 employees due to a severe cash crisis. the administrators told bbc news online that selling argonaut was the only way to save it as it had run out of cash. argonaut warned that it was low on cash 10 days ago when its shares were suspended from trading on the london stock exchange.  argonaut has been making games for some 18 years and is one the largest independent games developers in the uk.  along with its headquarters in north london  it operates studios in cambridge and sheffield. argonaut was behind the harry potter games which provided a healthy flow of cash into the company. but  like all software developers  argonaut needed a constant flow of deals with publishers. signs that it was in trouble emerged in august  when it warned it was heading for losses of £6m in the financial year due to delays in signing new contracts for games. those new deals were further delayed  leading argonaut to warn in mid-october that it was running out of cash and suspend trading of its shares on the london stock exchange. as part of cost-cutting measures  some 100 employees were fired.  when the news about the £6m loss came out  we knew there were going to be redundancies   said jason parkinson  one of the game developers sacked by argonaut.  a lot of people suspected that argonaut had been in trouble for some time   he told bbc news online. mr parkinson said staff were told the job losses were necessary to save argonaut from going under. at the start of the year  the company employed 268 people. after the latest round of cuts there are 80 staff at argonaut headquarters in edgware in north london  with 17 at its morpheme offices in kentish town  london  and 22 at the just add monsters base in cambridge.  argonaut called in administrators david rubin & partners on friday to find a way to rescue the company from collapse. it spent the weekend going over the company s finances and concluded that the only way to save the business was to put it up for sale. the administrator told bbc news online that the costs of restructuing would be too high  partly because of the overheads from the company s four premises across the uk. it said it was hopeful that it could save some 110 jobs by selling the business  saying it had had expressions of interest from several quarters and were looking for a quick sale. the administrator said it would ensure that staff made redundant would receive any wages  redundancy or holiday pay due to them  hopefully by christmas."

print(logreg.predict([complaint]))


# In[117]:


df[['Category','Category_id']].drop_duplicates()


# #Using the same data set, we are going to try some advanced techniques such as word embedding and neural networks. Now, let’s try some complex features than just simply counting words.
# 
# 

# **Decision Tree**
#     
#     Most powerful tools of supervised learning algorithms used for both classification and regression tasks.
#     DecisionTreeClassifier is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, …, K-1]) classification.

# In[68]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

#Convert it to Numerical, as Decision Tree expects Numerical input
X=cv.fit_transform(df['Article']).toarray()
y=cv.fit_transform(df['Category']).toarray()
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

model=DecisionTreeClassifier(criterion='gini')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


print('accuracy %s' % accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=df['Category'].unique()))
conf_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))# As are taking it in array form


# In[69]:


#plot_tree(model,filled=True,max_depth=2)


# In[70]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

#Convert it to Numerical, as Decision Tree expects Numerical input
X=cv.fit_transform(df['Article']).toarray()
y=cv.fit_transform(df['Category']).toarray()
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

RFC=RandomForestClassifier(random_state=7,criterion='gini')
RFC.fit(X_train,y_train)
y_pred = RFC.predict(X_test)


print('accuracy %s' % accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=df['Category'].unique()))
conf_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))# As are taking it in array form


# In[71]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean')

X=cv.fit_transform(df['Article']).toarray()
y=cv.fit_transform(df['Category']).toarray()

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print('accuracy %s' % accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=df['Category'].unique()))
conf_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))# As are taking it in array form


# In[24]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape,X.shape,y.shape


# In[23]:


df['Category'].unique()


# #Best Accuracy came with Logistic Regression so far > 95%.

# In[71]:





# #Use Pre-trained Vectors - Word2Vec

# In[ ]:


import gensim
#from gensim.models import word2vec
import gensim.downloader as api
print(list(gensim.downloader.info()['models'].keys()))


# In[ ]:


wv=api.load('fasttext-wiki-news-subwords-300')
#wv.save('path')


# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words=['one','two','man','woman','table']
sample_vectors=np.array([wv[word] for word in words])
pca=PCA(n_components=2)
result=pca.fit_transform(sample_vectors)
print(result)
plt.scatter(result[:,0],result[:,1])
for i,word in enumerate(words):
  plt.annotate(word,xy=(result[i,0],result[i,1]))


# In[ ]:


def sent_vec(sent):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res/ctr
    return wv_res


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)
    #print(doc,type(doc))
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[ ]:





# In[ ]:


sent_vec("I am happy")
import spacy
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
#print(stop_words)

import string
punctuations = string.punctuation
print(punctuations)
df['tokens'] = df['Article'].apply(spacy_tokenizer)
df['vectorized_Article'] = df['tokens'].apply(sent_vec)

df.head()


# In[ ]:


X = df['vectorized_Article'].to_list()
y = df['Category'].to_list()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
LogisticRegression()
from sklearn import metrics
predicted = classifier.predict(X_test)
print('accuracy %s' % accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))
conf_matrix(y_test, predicted)


# In[ ]:


complaint = "ocean s twelve raids box office ocean s twelve  the crime caper sequel starring george clooney  brad pitt and julia roberts  has gone straight to number one in the us box office chart.  it took $40.8m (£21m) in weekend ticket sales  according to studio estimates. the sequel follows the master criminals as they try to pull off three major heists across europe. it knocked last week s number one  national treasure  into third place. wesley snipes  blade: trinity was in second  taking $16.1m (£8.4m). rounding out the top five was animated fable the polar express  starring tom hanks  and festive comedy christmas with the kranks.  ocean s twelve box office triumph marks the fourth-biggest opening for a december release in the us  after the three films in the lord of the rings trilogy. the sequel narrowly beat its 2001 predecessor  ocean s eleven which took $38.1m (£19.8m) on its opening weekend and $184m (£95.8m) in total. a remake of the 1960s film  starring frank sinatra and the rat pack  ocean s eleven was directed by oscar-winning director steven soderbergh. soderbergh returns to direct the hit sequel which reunites clooney  pitt and roberts with matt damon  andy garcia and elliott gould. catherine zeta-jones joins the all-star cast.  it s just a fun  good holiday movie   said dan fellman  president of distribution at warner bros. however  us critics were less complimentary about the $110m (£57.2m) project  with the los angeles times labelling it a  dispiriting vanity project . a milder review in the new york times dubbed the sequel  unabashedly trivial ."
complaint_token = spacy_tokenizer(complaint)
#complaint_token

X_pred=sent_vec(complaint_token).reshape(1,-1)
#X_pred

classifier.predict(X_pred)

