import numpy as np
import os
from random import shuffle
import re
import sys
import urllib.request
import zipfile
import lxml.etree
import collections
from collections import Counter
import itertools
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init


# a Bi-directional RNN text classifier that predicts a TED talk is technology, entertainment, design, or none of these

# download dataset
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
texts = doc.xpath('//content/text()')
labels = doc.xpath('//head/keywords/text()')
del doc

# preprocess the texts: lowercase, remove text in parentheses, remove punctuation, tokenize into words (split on whitespace)
#removing text in parentheses
input_texts = [re.sub(r'\([^)]*\)', '', input_text) for input_text in texts]
#lowercase
input_texts = [input_text.lower() for input_text in input_texts]
#remove punctuation
input_texts = [re.sub(r'[^a-z0-9]+', ' ', input_text) for input_text in input_texts]
#tokenize into words
input_texts = [input_text.split() for input_text in input_texts]

#get list of all words, and feed them into a Counter
all_words = [word for input_text in input_texts for word in input_text]
print("There are {} tokens in the dataset.".format(len(all_words)))
all_words_counter = collections.Counter(all_words)

#remove some noise, take away the 100 most common and all words that only appear once
most_common_100 = [word for word, count in all_words_counter.most_common(100)]
only_once = [word for word, count in all_words_counter.most_common() if count == 1]
only_twice = [word for word, count in all_words_counter.most_common() if count == 2]
print("There are {} tokens that appear only once.".format(len(only_once)))
print("There are {} tokens that appear only twice.".format(len(only_twice)))
print(only_once[:10])
print(only_twice[:10])

to_remove = set(only_once + most_common_100)
print("There are {} unique tokens to remove.".format(len(to_remove)))
input_texts = [[word for word in input_text if word not in to_remove] for input_text in input_texts]
new_all_words = [word for input_text in input_texts for word in input_text]

'''
#remove all inputs that have less than a number tokens in them
inputs = zip(input_texts, labels)
inputs = [text_and_labels for text_and_labels in inputs if len(text_and_labels[0]) > 200]
print("There are now only {} inputs left.".format(len(inputs)))
input_texts, labels = zip(*inputs)
input_texts, labels = list(input_texts), list(labels)
'''

#truncating every text to only the first l_max tokens
l_max = 400
input_texts = [text[:l_max] for text in input_texts]
input_texts = [(['<zero_pad>'] * (l_max - len(text)) + text) for text in input_texts]

# creating the unique vocabulary lookup
vocab_list = list(set([word for input_text in input_texts for word in input_text]))
word_to_index = {}
index_to_word = {}
for i, word in enumerate(vocab_list):
    word_to_index[word] = i
    index_to_word[i] = word
input_indices_list = []
for input_text in input_texts:
    input_indices_list.append([word_to_index[word] for word in input_text])

#load glove word vectors
glove = KeyedVectors.load_word2vec_format('glove.6B.50d.w2vformat.txt', binary=False)

#creating embeddings, checking for each word in the input texts whether it is part of 
#the glove corpus, if yes intialize that row in the embeddings with the glove value, if
#not initialize it uniformly between [-.1, .1]
voc_len = len(word_to_index)
print("vocabulary size: {} words".format(voc_len))
counter = 0
not_found_list = []
embeddings = np.random.uniform(-.1, .1, size=(voc_len, 50))
for word, index in word_to_index.items():
    if word in glove.vocab:
        counter += 1
        embeddings[index] = glove[word]
    elif word == '<zero_pad>':
        embeddings[index] = np.zeros(50)
    else:
        not_found_list.append(word)
print("found {} word vectors, {} of our vocabulary".format(counter, float(counter)/voc_len))
#print("missing words e.g. {}".format(not_found_list[0:50]))

input_texts_embedding = np.zeros((len(input_texts), l_max, 50))
for i in range(0, len(input_texts)):
    for pos, index in enumerate(input_indices_list[i]):
        input_texts_embedding[i, pos, :]= embeddings[index]

print('input matrix done')

# process labels
label_lookup = ['ooo', 'Too', 'oEo', 'ooD', 'TEo', 'ToD', 'oED', 'TED']
for i in range(len(labels)):
    ted_labels = ['o', 'o', 'o']
    keyword_list = labels[i].split(', ')
    if 'technology' in keyword_list:
        ted_labels[0] = 'T'
    if 'entertainment' in keyword_list:
        ted_labels[1] = 'E'
    if 'design' in keyword_list:
        ted_labels[2] = 'D'
    labels[i] = ''.join(ted_labels)
    labels[i] = label_lookup.index(labels[i])

print('input labels done')

# turn into tensor
labels = np.asarray(labels)
labels= labels.astype(np.int64)
input_texts_embedding = input_texts_embedding.astype(np.float32)

labels = torch.from_numpy(labels)
input_texts_embedding = torch.from_numpy(input_texts_embedding)

# seperate data into train/val/test
train_dataset = data_utils.TensorDataset(input_texts_embedding[:1585,:,:], labels[:1585])
val_dataset = data_utils.TensorDataset(input_texts_embedding[1585:1835,:,:], labels[1585:1835])
train_val_dataset = data_utils.TensorDataset(input_texts_embedding[:1835,:,:], labels[:1835])
test_dataset = data_utils.TensorDataset(input_texts_embedding[1835:,:,:], labels[1835:])

print('dataset splitting done')
# model 

# Hyper Parameters
input_size = 50
hidden_size = 64
num_layers = 2
num_classes = 8
batch_size = 100
num_epochs = 20
learning_rate = 0.003


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


train_val_loader = torch.utils.data.DataLoader(dataset=train_val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)
                                           
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection 
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode hidden state of last time step
        out = out[:, -1, :]
        # avg of hidden states 
        #out = torch.mean(out, 1, True)
        out = self.fc(out)
        return out

rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)

rnn.load_state_dict(torch.load('model.pkl'))

# Test the Model
correct = 0
total = 0
for text, labels in train_loader:
    text = Variable(text)
    outputs = rnn(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on train data: %d %%' % (100 * correct / total))

# Test the Model
correct = 0
total = 0
for text, labels in val_loader:
    text = Variable(text)
    outputs = rnn(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on val data: %d %%' % (100 * correct / total))

# Test the Model
correct = 0
total = 0
for text, labels in test_loader:
    text = Variable(text)
    outputs = rnn(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on test data: %d %%' % (100 * correct / total))