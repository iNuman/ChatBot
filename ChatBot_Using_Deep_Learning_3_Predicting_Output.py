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

model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
model.save("Model.tflearn")
# Starting Predictions
"""
Now its time to actually use the model! Ideally we want to generate a response 
to any sentence the user types in. To do this we need to remember that our model 
does not take string input, it takes a bag of words. We also need to realize that 
our model does not spit out sentences, it generates a list of probabilities for all 
of our classes. This makes the process to generate a response look like the following:
– Get some input from the user
– Convert it to a bag of words
– Get a prediction from the model
– Find the most probable class
– Pick a response from that class
"""

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat():
    print("Hei there it's Bot! (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7: # pick the largest index of the number
	        for tg in data["intents"]:
	            if tg['tag'] == tag:
	                responses = tg['responses']

	        print("Bot: ", random.choice(responses))

        else:
        	print("Bot: Sorry I ran out of Words :(")

        """
        The bag_of_words function will transform our string input to a bag of words
        using our created words list. The chat function will handle getting a prediction 
        from the model and grabbing an appropriate response from our JSON file of responses.
        """

chat()
