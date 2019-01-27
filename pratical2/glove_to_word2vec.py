from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B/glove.6B.50d.txt'
word2vec_output_file = 'glove.6B.50d.w2vformat.txt'
glove2word2vec(glove_input_file, word2vec_output_file)