#!/usr/bin/env python
# coding: utf-8

# ## ChatBot

# In[1]:


import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn
import random
import json


# ## Reading file

# In[2]:


with open("intents.json") as file:
    data = json.load(file)
    
# print(data["intents"])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        '''
        Now we'll do stemming! 
        Stemming means it will look through each word and will remove extra works e.g whats up will be change to what up
        and Is anyone there? from here ? will be removed
        this will remove extra characters like these because during training we only care about main meaning of words
        and thus this will make our model more efficient
        '''
        
        wrds = nltk.word_tokenize(pattern) # since this is allready a list and we can add two list using extend keyword
        words.extend(wrds) # combined both lists
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words))) # set will remove duplicates, list will convert words back to list and sort will sort list

labels = sorted(labels)

training = []
output = []

output_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    
    wrds_exist = [stemmer.stem(w) for w in doc]
    
    for w in words:
        if w in wrds_exist: # if word exists in the word list that we stemmed above then appen 1 with the bag
            bag.append(1)
        else:
            bag.append(0)
           
    output_row = output_empty[:]
    output_row[labels.index(docs_y[x])] = 1 # we'll look through labels list where that value is and will set that value to 1
    training.append(bag)
    output.append(output_row)

    
'''
Now We're gona convert training and output lists to numpy array
'''
training = np.array(training)
output = np.array(output)
    
    
    

        


# In[ ]:


tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])]) # input layer
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net, 8) # hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # activation function along wiht output
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=10000, batch_size=8, show_metric=True)
model.save("Model.tflearn")


# In[ ]:





# In[ ]:




