# TED-Talks-Text-Classification
## Introduction
In 2017, I followed the deep learning NLP course at Oxford University and finished the praticals. The course materials can be found [here](https://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/). The praticals was about applying natural language processing (NLP) techniques to a [real world dataset](https://wit3.fbk.eu/mono.php?release=XML_releases&tinfo=cleanedhtml_ted): TED talks scripts. 

## Word embeddeing
In the first pratical, I trained the word2vec embedding models using Wikipedia data and the TED talks data. After training the models, I also analyzed and visualized the learned embeddings using t-SNE. 

## Text classification using MLP
The second pratical was implementing a multi-class text classification model using Multilayer Perceptron (MLP) for the TED talks dataset. "TED" stands for "technology", "entertainment", and "design". The model classifies each document into one of the 8 classes: "T", "E", "D", "TE", "TD", "ED", "TED" and "None".

## Text classification using Bi-LSTM with attention
The last pratical is an extention of the second pratical. It includes more advanced classification models including RNNs and LSTMs. I also implemented a Bi-LSTM model with attention mechanism, which is the state-of-the-art and the go-to choice for language tasks at the time. 
