## ChatBot
A Simple ChatBot using Deep Learning Neural Network

## Screenshot
<p align="center">
  <img src="https://github.com/iNuman/ChatBot/blob/master/ChatBot.gif" width="900" height="370">
</p>

## Pre-Requisites
I faced some issues during implementaiton of **tfLearn** using `python 3.7` and it worked for me with `python 3.65`
## Packages

Let's first import all the packages that you will need during this project.
- [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
- [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis. 
- [nltk](https://www.nltk.org/) is a package to work with human language data.
- [tensorflow](https://www.tensorflow.org/) is an open source high level library, using data flow graphs to build models.
- [tflearn](http://tflearn.org/) is a deep learning library built on top of Tensorflow.
- [Pickle](https://docs.python.org/3/library/pickle.html) it's pre-built python libraries used for `Saving` and `loading` models.
- [json](https://docs.python.org/2/library/json.html) Refer to Documentation

### Imports

```python
    import nltk
    from nltk.stem.lancaster import LancasterStemmer
    stemmer = LancasterStemmer()
    import numpy as np
    import tensorflow as tf
    import tflearn
    import random
    import json
```


### Implementation tflearn:
```python
    tf.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])]) # input layer, 
    net = tflearn.fully_connected(net, 8) # hidden layer
    net = tflearn.fully_connected(net, 8) # hidden layer
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # activation function along with output
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
```

## Clonning
```yaml

            git clone https://github.com/iNuman/ChatBot                
```

## Contact
<p align="left">
<ul style="list-style-type:circle;">
  <li>LinkedIn  : <a href="https://www.linkedin.com/in/-inuman/">https://www.linkedin.com/in/-inuman/</a>
  <li>Instagram : <a href="https://instagram.com/inoumn">https://instagram.com/inoumn</a></li>
  <li>Facebook  : <a href="https://www.facebook.com/iNuman51">https://www.facebook.com/iNuman51</a></li>
</ul></p>


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/iNuman/ChatBot/blob/master/LICENSE.md)

```
Copyright (c) 2020 Numan Ali

    Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be 
included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
```

