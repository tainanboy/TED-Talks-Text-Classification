import numpy as np
import os
from random import shuffle
import re
import sys
import urllib.request
import zipfile
import lxml.etree
import collections
from collections import Counter, defaultdict
import itertools
from gensim.models import Word2Vec, KeyedVectors
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hyper Parameters
l_max = 500
input_size = 200
hidden_size = 100
num_layers = 1
num_classes = 8
batch_size = 100
num_epochs = 30
learning_rate = 0.0003

def removeParenthesis(texts):
    # texts is a list of docs
    input_texts = [re.sub(r'\([^)]*\)', '', input_text) for input_text in texts]
    return input_texts

def lowercase(input_texts):
    input_texts = [input_text.lower() for input_text in input_texts]
    return input_texts

def removePuntuation(input_texts):
    input_texts = [re.sub(r'[^a-z0-9]+', ' ', input_text) for input_text in input_texts]
    return input_texts

def tokenizePunctuation(input_texts):
    input_texts = [re.sub(r'([^a-z0-9\s])', r' <\1_token> ', input_text) for input_text in input_texts]
    return input_texts

def tokenizer(input_texts):
    input_texts = [input_text.split() for input_text in input_texts]
    return input_texts

def keyword_checker(l):
    keywords = ['technology', 'tech', 'entertainment', 'design']
    for word in l:
        if word in keywords:
            print("The word {} cannot be removed.".format(word))
            l.remove(word)
    return l 

def word_counter(input_texts):
    all_words = [word for input_text in input_texts for word in input_text]
    print("There are {} tokens in the dataset.".format(len(all_words)))
    all_words_counter = collections.Counter(all_words)
    return all_words_counter

def most_common_k(all_words_counter, k):
    most_common_100 = [word for word, count in all_words_counter.most_common(k)]
    return most_common_100

def appear_k_times(all_words_counter, k):
    only_k = [word for word, count in all_words_counter.most_common() if count == k]
    return only_k

def remove_small_docs(input_texts, labels, k):
    inputs = zip(input_texts, labels)
    inputs = [text_and_labels for text_and_labels in inputs if len(text_and_labels[0]) > k]
    print("There are now only {} inputs left.".format(len(inputs)))
    input_texts, labels = zip(*inputs)
    input_texts, labels = list(input_texts), list(labels)
    return input_texts, labels

def truncate(input_texts, l_max):
    input_texts = [text[:l_max] for text in input_texts]
    return input_texts

def padding(input_texts, l_max):
    input_texts = [(['<zero_pad>'] * (l_max - len(text)) + text) for text in input_texts]
    return input_texts

def create_vocabulary(input_texts):
    vocab_list = list(set([word for input_text in input_texts for word in input_text]))
    word_to_index = {}
    index_to_word = {}
    for i, word in enumerate(vocab_list):
        word_to_index[word] = i
        index_to_word[i] = word
    input_indices_list = []
    for input_text in input_texts:
        input_indices_list.append([word_to_index[word] for word in input_text])
    return word_to_index, index_to_word, input_indices_list

def plot_length_his(input_texts):
    #histogram over input lengths
    Y_plot, X_plot = np.histogram([len(text) for text in input_texts], bins=10)
    X_plot = np.arange(10)
    plt.bar(X_plot, +Y_plot, facecolor='#9999ff', edgecolor='white')
    plt.savefig('doclength.jpg')
    plt.close()

def label_his(labels, filestring):
    #plotting a histogram over the label distribution in the entire dataset
    Y_plot = np.histogram(labels, bins=8)[0]
    X_plot = np.arange(8)
    plt.bar(X_plot, +Y_plot, facecolor='#9999ff', edgecolor='white')
    for x,y in zip(X_plot,Y_plot):
        plt.text(x, y+0.05, label_lookup[x], ha='center', va= 'bottom')
    plt.savefig(filestring)
    plt.close()

def split_data(input_texts_embedding, labels, labels_index_dict, method='keepd'):
    if method == 'keepd':
        train_data = []
        val_data = []
        test_data = []
        train_labels = []
        val_labels = []
        test_labels = []
        for k in labels_index_dict.keys():
            train_num = int(len(labels_index_dict[k])*0.8)
            val_num = int(len(labels_index_dict[k])*0.1)
            shuffle(labels_index_dict[k])
            train_i = labels_index_dict[k][:train_num]
            val_i = labels_index_dict[k][train_num:train_num+val_num]
            test_i = labels_index_dict[k][train_num+val_num:]
            for i in train_i:
                train_data.append(input_texts_embedding[i,:,:])
                train_labels.append(labels[i])
            for i in val_i:
                val_data.append(input_texts_embedding[i,:,:])
                val_labels.append(labels[i])
            for i in test_i:
                test_data.append(input_texts_embedding[i,:,:])
                test_labels.append(labels[i])
        # see distribution 
        label_his(train_labels, 'trainlabel.jpg')
        label_his(val_labels, 'vallabel.jpg')
        label_his(test_labels, 'testlabel.jpg')
        #
        train_data = np.asarray(train_data, dtype=np.float32)
        train_data = torch.from_numpy(train_data)
        val_data = np.asarray(val_data, dtype=np.float32)
        val_data = torch.from_numpy(val_data)
        test_data = np.asarray(test_data, dtype=np.float32)
        test_data = torch.from_numpy(test_data)
        train_labels = np.asarray(train_labels, dtype=np.int64)
        train_labels = torch.from_numpy(train_labels)
        val_labels = np.asarray(val_labels, dtype=np.int64)
        val_labels = torch.from_numpy(val_labels)
        test_labels = np.asarray(test_labels, dtype=np.int64)
        test_labels = torch.from_numpy(test_labels)
        #
        train_dataset = data_utils.TensorDataset(train_data, train_labels)
        val_dataset = data_utils.TensorDataset(val_data, val_labels)
        #train_val_dataset = data_utils.TensorDataset(torch.cat(train_data,val_data), torch.cat(train_data,val_data))
        test_dataset = data_utils.TensorDataset(test_data, test_labels)
        print('dataset splitting done')
        return train_dataset, val_dataset, test_dataset
    else:
        # turn into tensor
        labels = np.asarray(labels, dtype=np.int64)
        #labels= labels.astype(np.int64)
        input_texts_embedding = input_texts_embedding.astype(np.float32)

        labels = torch.from_numpy(labels)
        input_texts_embedding = torch.from_numpy(input_texts_embedding)
        # see distribution 
        label_his(labels[:1585], 'trainlabel.jpg')
        label_his(labels[1585:1835], 'vallabel.jpg')
        label_his(labels[:1835], 'testlabel.jpg')
        # splitting
        train_dataset = data_utils.TensorDataset(input_texts_embedding[:1585,:,:], labels[:1585])
        val_dataset = data_utils.TensorDataset(input_texts_embedding[1585:1835,:,:], labels[1585:1835])
        #train_val_dataset = data_utils.TensorDataset(input_texts_embedding[:1835,:,:], labels[:1835])
        test_dataset = data_utils.TensorDataset(input_texts_embedding[1835:,:,:], labels[1835:])
        print('dataset splitting done')
        return train_dataset, val_dataset, test_dataset

def pytorch_plot_losses(softmax_loss_history=None, mse_loss_history=None, 
                                        test_losses_softmax=None, test_losses_mse=None):
    plt.clf()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if softmax_loss_history:
        ax1.plot(softmax_loss_history, color="blue")
    if test_losses_softmax:
        ax1.plot(test_losses_softmax, color="green")
    ax2 = ax1.twinx()
    if mse_loss_history:
        ax2.plot(mse_loss_history, color="red")
    if test_losses_mse:
        ax2.plot(test_losses_mse, color="black")
    #ax2.set_yscale('log')
    plt.savefig('output_losses.png')

def ans_distribution(collect_preds, collect_truth):
    plt.clf()
    bins = np.arange(9)
    plt.hist(np.array(collect_preds), bins, alpha=0.5, label='predictions', color = "red")
    plt.hist(np.array(collect_truth), bins, alpha=0.5, label='truth', color = "blue")
    plt.legend(loc='upper right')
    plt.savefig('ansdistribution.jpg')

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

# preprocess the texts: lowercase, replace punctuation with tokens, tokenize into words (split on whitespace)
#lowercase
input_texts = lowercase(texts)
#replace punctuation with punctuation tokens
input_texts = tokenizePunctuation(input_texts)
#tokenize into words
input_texts = tokenizer(input_texts)

#get list of all words, and feed them into a Counter
all_words_counter = word_counter(input_texts)

#remove some noise, take away the 100 most common and all words that only appear once
most_common_100 = most_common_k(all_words_counter, 150)
only_once = appear_k_times(all_words_counter, 1)
only_twice = appear_k_times(all_words_counter, 2)
only_3 = appear_k_times(all_words_counter, 3)
only_4 = appear_k_times(all_words_counter, 4)
only_5 = appear_k_times(all_words_counter, 5)

print("There are {} tokens that appear only once.".format(len(only_once)))
print("There are {} tokens that appear only twice.".format(len(only_twice)))
print("There are {} tokens that appear only twice.".format(len(only_3)))
print("There are {} tokens that appear only twice.".format(len(only_4)))
print("There are {} tokens that appear only twice.".format(len(only_5)))

most_common_100 = keyword_checker(most_common_100)
only_once = keyword_checker(only_once)
only_twice = keyword_checker(only_twice)

to_remove = set(only_once + only_twice + only_3 + only_4 + most_common_100)
print("There are {} unique tokens to remove.".format(len(to_remove)))
input_texts = [[word for word in input_text if word not in to_remove] for input_text in input_texts]
new_all_words = [word for input_text in input_texts for word in input_text]

#remove all inputs that have less than a number tokens in them
#input_texts, labels = remove_small_docs(input_texts, labels, 200)

# see input docs length distribution
plot_length_his(input_texts)

#truncating every text to only the first l_max tokens
input_texts = truncate(input_texts, l_max)
input_texts = padding(input_texts, l_max)

#creating the unique vocabulary lookup
word_to_index, index_to_word, input_indices_list = create_vocabulary(input_texts)

#load glove word vectors
if input_size == 50:
    glove = KeyedVectors.load_word2vec_format('glove.6B.50d.w2vformat.txt', binary=False)
elif input_size == 100:
    glove = KeyedVectors.load_word2vec_format('glove.6B.100d.w2vformat.txt', binary=False)
elif input_size == 200:
    glove = KeyedVectors.load_word2vec_format('glove.6B.200d.w2vformat.txt', binary=False)
elif input_size == 300:
    glove = KeyedVectors.load_word2vec_format('glove.6B.300d.w2vformat.txt', binary=False)

#creating embeddings, checking for each word in the input texts whether it is part of 
#the glove corpus, if yes intialize that row in the embeddings with the glove value, if
#not initialize it uniformly between [-.1, .1]
voc_len = len(word_to_index)
print("vocabulary size: {} words".format(voc_len))
counter = 0
not_found_list = []
embeddings = np.random.uniform(-.3, .3, size=(voc_len+1, input_size))
for word, index in word_to_index.items():
    if word in glove.vocab:
        counter += 1
        embeddings[index] = glove[word]
print("found {} word vectors, {} of our vocabulary".format(counter, float(counter)/voc_len))

input_texts_embedding = np.zeros((len(input_texts), l_max, input_size))
for i in range(0, len(input_texts)):
    for pos, index in enumerate(input_indices_list[i]):
        input_texts_embedding[i, pos, :]= embeddings[index]

print('input matrix done')

# process labels
labels_index_dict = defaultdict(list)
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
    labels_string = ''.join(ted_labels)
    labels_index_dict[label_lookup.index(labels_string)].append(i)
    labels[i] = label_lookup.index(labels_string)
#print(Counter(labels))
#print(labels_index_dict.keys())
print('input labels done')

# seperate data into train/val/test
train_dataset, val_dataset, test_dataset = split_data(input_texts_embedding, labels, labels_index_dict)

# model 
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

'''
train_val_loader = torch.utils.data.DataLoader(dataset=train_val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
'''

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)
                                           
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
                            batch_first=True, dropout = 0.2, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection 
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        #out = out[:, -1, :]
        # avg of hidden states 
        out = torch.mean(out, 1)
        out = self.fc(out)
        return out

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout = 0.3, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        # Decode hidden state of last time step
        #out = out[:, -1, :]
        # avg of hidden states 
        out = torch.mean(out, 1)
        #
        out = self.dropout(out)
        out = self.fc(out)
        return out

class BiattGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiattGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout = 0.3, bidirectional=True)
        self.attu = nn.Linear(hidden_size*2, hidden_size*2) 
        self.atts = nn.Linear(hidden_size*2, 1, bias=False) 
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.batch_norm = nn.BatchNorm1d(hidden_size*2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # 2 for bidirection 
        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        # Decode hidden state of last time step
        #out = out[:, -1, :]
        # avg of hidden states 
        #out = torch.mean(out, 1)
        # attention layer
        alpha = Variable(torch.Tensor(out.size(0), out.size(1)))
        for t in range(out.size(1)):
            ht = out[:, t, :]
            ut = self.attu(ht)
            ut = self.batch_norm(ut)
            ut = self.tanh(ut)
            s = self.atts(ut)
            alpha[:, t] = s
        norm_alpha = self.softmax(alpha)
        norm_alpha = torch.unsqueeze(norm_alpha, 2)
        out = torch.transpose(out, 1, 2)
        out = torch.bmm(out, norm_alpha)
        out = torch.squeeze(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

rnn = BiattGRU(input_size, hidden_size, num_layers, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    
# Train the Model 
iter_losses_softmax = []
epoch_losses_softmax = []
viter_losses_softmax = []
test_losses_softmax = []

for epoch in range(num_epochs):
    for i, (text, labels) in enumerate(train_loader):
        rnn.train(True)
        text = Variable(text)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(text)
        loss = criterion(outputs, labels)
        iter_losses_softmax.append(loss.data.numpy()[0])
        loss.backward()
        optimizer.step()
        
        if (i+1) % 4 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    
    for i, (vtext, vlabels) in enumerate(val_loader):
        rnn.train(False)
        vtext = Variable(vtext)
        vlabels = Variable(vlabels)
        voutputs = rnn(vtext)
        vloss = criterion(voutputs, vlabels)
        viter_losses_softmax.append(vloss.data.numpy()[0])
        
    epoch_losses_softmax.append(np.mean(iter_losses_softmax))
    test_losses_softmax.append(np.mean(viter_losses_softmax))

pytorch_plot_losses(softmax_loss_history=epoch_losses_softmax, test_losses_softmax=test_losses_softmax)


# Save the Model
torch.save(rnn.state_dict(), 'model.pkl')

# Test the Model
correct = 0
total = 0
for text, labels in train_loader:
    rnn.train(False) 
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
    rnn.train(False) 
    text = Variable(text)
    outputs = rnn(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on val data: %d %%' % (100 * correct / total))

# Test the Model
correct = 0
total = 0
collect_preds = []
collect_truth = []
for text, labels in test_loader:
    rnn.train(False) 
    text = Variable(text)
    outputs = rnn(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    for l in predicted.numpy():
        collect_preds.append(int(l))
    for l in labels.numpy():
        collect_truth.append(int(l))
print('Accuracy of the network on test data: %d %%' % (100 * correct / total))

ans_distribution(collect_preds, collect_truth)

