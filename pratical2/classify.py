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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
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


#a text classifier that predicts a TED talk is technology, entertainment, design, or none of these
#using mean word vectors as document vector

# Hyper Parameters
l_max = 500 
input_size = 50
hidden_size1 = 100
hidden_size2 = 100
num_classes = 8
num_epochs = 200
batch_size = 50
learning_rate = 0.0003

#download dataset
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

#For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
texts = doc.xpath('//content/text()')
labels = doc.xpath('//head/keywords/text()')
del doc

#print(type(texts))
#print(texts[0])
#print(labels[0])
#print(len(texts))
#print(len(labels))

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
                train_data.append(input_texts_embedding[i,:])
                train_labels.append(labels[i])
            for i in val_i:
                val_data.append(input_texts_embedding[i,:])
                val_labels.append(labels[i])
            for i in test_i:
                test_data.append(input_texts_embedding[i,:])
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
        train_dataset = data_utils.TensorDataset(input_texts_embedding[:1585,:], labels[:1585])
        val_dataset = data_utils.TensorDataset(input_texts_embedding[1585:1835,:], labels[1585:1835])
        #train_val_dataset = data_utils.TensorDataset(input_texts_embedding[:1835,:], labels[:1835])
        test_dataset = data_utils.TensorDataset(input_texts_embedding[1835:,:], labels[1835:])
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

#preprocess the texts: lowercase, remove text in parentheses, remove punctuation, tokenize into words (split on whitespace)
input_texts = removeParenthesis(texts)
input_texts = lowercase(input_texts)
input_texts = removePuntuation(input_texts)
input_texts = tokenizer(input_texts)
#print(len(input_texts))
#print(input_texts[0][:50])

#get list of all words, and feed them into a Counter
all_words_counter = word_counter(input_texts)

#remove some noise, take away the 100 most common and all words that only appear once
most_common_100 = most_common_k(all_words_counter, 150)
only_once = appear_k_times(all_words_counter, 1)
only_twice = appear_k_times(all_words_counter, 2)
only_3 = appear_k_times(all_words_counter, 3)
only_4 = appear_k_times(all_words_counter, 4)
only_5 = appear_k_times(all_words_counter, 5)
only_6 = appear_k_times(all_words_counter, 6)
print("There are {} tokens that appear only once.".format(len(only_once)))
print("There are {} tokens that appear only twice.".format(len(only_twice)))
print("There are {} tokens that appear only twice.".format(len(only_3)))
print("There are {} tokens that appear only twice.".format(len(only_4)))
print("There are {} tokens that appear only twice.".format(len(only_5)))
print("There are {} tokens that appear only twice.".format(len(only_6)))


most_common_100 = keyword_checker(most_common_100)
only_once = keyword_checker(only_once)
only_twice = keyword_checker(only_twice)
only_3 = keyword_checker(only_3)
only_4 = keyword_checker(only_4)
only_5 = keyword_checker(only_5)
only_6 = keyword_checker(only_6)

to_remove = set(only_once + only_twice + only_3 + only_4 + most_common_100)
print("There are {} unique tokens to remove.".format(len(to_remove)))
input_texts = [[word for word in input_text if word not in to_remove] for input_text in input_texts]
new_all_words = [word for input_text in input_texts for word in input_text]

#remove all inputs that have less than a number tokens in them
#input_texts, labels = remove_small_docs(input_texts, labels, 200)

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
embeddings = np.random.uniform(-.1, .1, size=(voc_len, input_size))
for word, index in word_to_index.items():
    if word in glove.vocab:
        counter += 1
        embeddings[index] = glove[word]
    elif word == '<zero_pad>':
        embeddings[index] = np.zeros(input_size)
print("found {} word vectors, {} of our vocabulary".format(counter, float(counter)/voc_len))

input_texts_embedding = np.zeros((len(input_texts), input_size))
for i in range(0, len(input_texts)):
    vector_sum = np.zeros(input_size)
    for index in input_indices_list[i]:
        vector_sum = vector_sum+embeddings[index]
    input_texts_embedding[i, :] = vector_sum/len(input_indices_list[i])

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


# mlp model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        #init.kaiming_normal(self.fc1.weight, mode='fan_out')
        init.xavier_normal(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.batch_norm1(out)
        #out = self.tanh(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
    
net = Net(input_size, hidden_size1, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.005)

# Train the Model
iter_losses_softmax = []
epoch_losses_softmax = []
viter_losses_softmax = []
test_losses_softmax = []
for epoch in range(num_epochs):
    for i, (text, labels) in enumerate(train_loader): 
        net.train(True) 
        # Convert torch tensor to Variable
        text = Variable(text)
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(text)
        loss = criterion(outputs, labels)
        iter_losses_softmax.append(loss.data.numpy()[0])
        loss.backward()
        optimizer.step()
        
        if (i+1) % 30 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    for i, (vtext, vlabels) in enumerate(val_loader):
        net.train(False) 
        vtext = Variable(vtext)
        vlabels = Variable(vlabels)
        voutputs = net(vtext)
        vloss = criterion(voutputs, vlabels)
        viter_losses_softmax.append(vloss.data.numpy()[0])

    epoch_losses_softmax.append(np.mean(iter_losses_softmax))
    test_losses_softmax.append(np.mean(viter_losses_softmax))

pytorch_plot_losses(softmax_loss_history=epoch_losses_softmax, test_losses_softmax=test_losses_softmax)

# Test the Model
correct = 0
total = 0
for text, labels in train_loader:
    net.train(False)
    text = Variable(text)
    outputs = net(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy of the network on train data: %d %%' % (100 * correct / total))

# Test the Model
correct = 0
total = 0
for text, labels in val_loader:
    net.train(False) 
    text = Variable(text)
    outputs = net(text)
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
    net.train(False) 
    text = Variable(text)
    outputs = net(text)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    for l in predicted.numpy():
        collect_preds.append(int(l))
    for l in labels.numpy():
        collect_truth.append(int(l))
print('Accuracy of the network on test data: %d %%' % (100 * correct / total))

ans_distribution(collect_preds, collect_truth)

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
