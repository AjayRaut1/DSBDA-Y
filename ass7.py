#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip list')


# In[2]:


import nltk


# In[3]:


import nltk
nltk.download('punkt')


# In[4]:


from nltk import tokenize
from nltk.tokenize import sent_tokenize
text="Good Day Everyone,How are you all today?Its fun learning Data Analysis.Hope you all are practising well"
text


# In[5]:


tokenized_text=sent_tokenize(text)
print(tokenized_text)


# In[6]:


from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)


# In[7]:


from nltk.probability import FreqDist
fdist=FreqDist(tokenized_word)
print(fdist)


# In[8]:


fdist.most_common(4)


# In[9]:


import matplotlib.pyplot as plt

fdist.plot(30,cumulative=False)
plt.show()


# In[10]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[11]:


filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence :",tokenized_word)
print("Filtered Sentence:",filtered_sent)


# In[12]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
ps=PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("Filtered Sentence :",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[13]:


sent="Albert Einstein was born in Ulm,Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)


# In[14]:


nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(tokens)

