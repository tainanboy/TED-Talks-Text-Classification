import numpy as np
import os
from random import shuffle
import re
import sys
import urllib.request
import zipfile
import lxml.etree
from collections import Counter
import itertools
from gensim.models import Word2Vec

# part0: download dataset
# Download the dataset if it's not already there: this may take a minute as it is 75MB
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

# For now, we're only interested in the subtitle text, so let's extract that from the XML:
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
del doc

# part1: preprocessing
#i = input_text.find("Hyowon Gweon: See this?")
#input_text[i-20:i+150]
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)

sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
print(sentences_strings_ted[:5])

sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)

print(sentences_ted[1])
print(sentences_ted[2])

# part2: word frequences 
counts_ted_top1000 = []
flat_sentences_ted = list(itertools.chain(*sentences_ted))
print(len(flat_sentences_ted))
print(type(Counter(flat_sentences_ted)))
common_wordslist = Counter(flat_sentences_ted).most_common(1000)
counts_ted_top1000 = [i[1] for i in common_wordslist]
print(counts_ted_top1000[1])
print(common_wordslist[0])

# part3: train word2vec
sentences = sentences_ted
model_ted = Word2Vec(sentences, size=100, window=5, min_count=10, workers=4)
print(len(model_ted.wv.vocab))

# Part 4: Ted Learnt Representations
print(model_ted.most_similar("man"))
print(model_ted.most_similar("computer"))