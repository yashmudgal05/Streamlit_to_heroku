#!/usr/bin/env python
# coding: utf-8

# ## Emotion Recognition from Text Data 

# ### Data Preparation

# In[1]:


import numpy as np
from numpy import array
import pandas as pd
import pickle


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


import preprocess_pranjalSagar as ps


# In[4]:


df = pd.read_csv('text_to_emotion.csv')
df.sample(5)


# In[5]:


df.shape


# In[6]:


df['emotion'].value_counts()


# In[ ]:





# ### Preprocessing and Cleaning 

# In[7]:


get_ipython().run_cell_magic('time', '', "df['text'] = df['text'].apply(lambda x: str(x).lower())\ndf['text'] = df['text'].apply(lambda x: ps.cont_exp(x))\n\ndf['text'] = df['text'].apply(lambda x: ps.remove_special_chars(x))\ndf['text'] = df['text'].apply(lambda x: ps.remove_accented_chars(x))\n\n# df['text'] = df['text'].apply(lambda x: ps.make_base(x))\n# df['text'] = df['text'].apply(lambda x: ps.spelling_correctin(x).raw_sentences[0])")


# In[8]:


df['text']


# In[9]:


df.sample(5)


# In[ ]:





# In[10]:


df.sample(5)


# In[ ]:





# In[11]:


X = df['text']
y = df['emotion']


# In[12]:


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)


# In[13]:


X.shape, y.shape


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


# In[15]:


X.shape, y.shape, X_train.shape, X_test.shape


# In[16]:


# model creation
clf = LogisticRegression(solver = 'liblinear', multi_class='auto')


# In[ ]:





# In[17]:


# model training
clf.fit(X_train, y_train)


# In[18]:


# predicting output using the model
y_pred = clf.predict(X_test)


# In[19]:


# model accuracy
print(classification_report(y_test, y_pred))


# In[20]:


from mlxtend.plotting import plot_confusion_matrix


# In[21]:


# model accuracy
print(confusion_matrix(y_test, y_pred))


# In[22]:


plot_confusion_matrix(confusion_matrix(y_test, y_pred))


# In[ ]:





# In[23]:


pickle.dump(clf, open('model_emotionRecogination.pkl', 'wb'))


# In[24]:


pickle.dump(tfidf, open('tfidf_emotionRecogination.pkl', 'wb'))


# In[25]:


del clf


# In[26]:


del tfidf


# In[27]:


clf


# In[28]:


tfidf


# In[ ]:





# ### Predict Text Emotion with Custom Data 

# In[ ]:





# In[29]:


# load model from local computer
model_logisticRegression = pickle.load(open('model_emotionRecogination.pkl', 'rb'))
tfidf_logisticRegression = pickle.load(open('tfidf_emotionRecogination.pkl', 'rb'))


# In[30]:


x = ['i am so happy. thanks a lot']


# In[31]:


emotion = model_logisticRegression.predict(tfidf_logisticRegression.transform(x))
emotion


# In[32]:


emotion[0]


# In[ ]:





# In[ ]:




