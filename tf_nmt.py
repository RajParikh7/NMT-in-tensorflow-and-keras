# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:22:40 2018

@author: rajpa
"""

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
#from attention_wrap import Attention
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# load doc into memory
tf.reset_default_graph()

def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')

# load a clean dataset

from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:9000], dataset[9000:]
# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')
# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

embedding_size=256
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

def max_length(lines):
	return max(len(line.split()) for line in lines)
# fit a tokenizer

def length(lines):
	return [len(line.split()) for line in lines]

def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
import numpy as np
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
inputs_encoder=[]
inputs=train[:,1]
outputx=train[:,0]
num_units=256
outputs1=[]
for line in outputx:
    outputs1.append("<s> "+line + " </s>")
    
outputs2=[]
for line in outputx:
    outputs2.append(line+" </s>")
    
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
    X = np.array(tokenizer.texts_to_sequences(lines))
	# pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X.transpose() 




#total timestamps
inp_len=max_length(inputs)
out_len=max_length(outputs2)
batch_size=9000
learning_rate=0.01
#encoder size and data
ger_tokenizer = create_tokenizer(dataset[:, 1])
src_vocab_size = len(ger_tokenizer.word_index) + 1
inputs_encoder = encode_sequences(ger_tokenizer, inp_len, train[:, 1])
encoder_inputs = tf.placeholder(shape=(inp_len,None),dtype=tf.int32)

#decoder size and data
eng_tokenizer = create_tokenizer(outputs1)
tar_vocab_size = len(eng_tokenizer.word_index) + 1

inputs_decoder = encode_sequences(eng_tokenizer, out_len, outputs1)
output_decoder= encode_sequences(eng_tokenizer, out_len, outputs2)
output_decoder= output_decoder.transpose() 
decoder_inputs = tf.placeholder(shape=(out_len,None),dtype=tf.int32)
decoder_output = tf.placeholder(shape=(None,out_len),dtype=tf.int32)
seq_len_inp=tf.placeholder(shape=(None),dtype=tf.int32)
seq_len_out=tf.placeholder(shape=(None),dtype=tf.int32)
input_len_vec=[len(line.split()) for line in inputs]
output_len_vec=[len(line.split()) for line in outputs2]

# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]

#embedding vector for encoder
embedding_encoder = tf.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size])

encoder_emb_inp = tf.nn.embedding_lookup(
    embedding_encoder, encoder_inputs)

#embedding vector for decoder
embedding_decoder = tf.get_variable(
    "embedding_decoder", [tar_vocab_size, embedding_size])

decoder_emb_inp = tf.nn.embedding_lookup(
    embedding_decoder, decoder_inputs)

#rnn_cell = tf.nn.rnn_cell.BasicRNNCell(256)
#initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    time_major=True,dtype=tf.float32)

attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

# Create an attention mechanism
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units, attention_states,
    memory_sequence_length=input_len_vec)


decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

attn_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)

initial_state = attn_cell.zero_state( batch_size = batch_size , dtype=tf.float32 )
initial_state = initial_state.clone(cell_state = encoder_state)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, sequence_length=output_len_vec, time_major=True)
# Decoder
projection_layer = tf.layers.Dense(
    tar_vocab_size, use_bias=False)

decoder = tf.contrib.seq2seq.BasicDecoder(
    attn_cell, helper, initial_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs,_,__ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=decoder_output, logits=logits)
train_loss = (tf.reduce_sum(crossent) /
    batch_size)

max_gradient_norm=5

params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients,max_gradient_norm)
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
#    update_step = optimizer.apply_gradients(
#   zip(clipped_gradients, params))


# Create a summary to monitor cost tensor
display_epoch=10
# Start training
init = tf.global_variables_initializer()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    for epoch in range(10):
        avg_cost = 0.
        
        # Loop over all batches
        _, c = sess.run([optimizer, train_loss],
                                     feed_dict={encoder_inputs: inputs_encoder, decoder_inputs: inputs_decoder,decoder_output:output_decoder})
            # Write logs at every iteration
            #summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
        avg_cost=c  
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

#inference
tgt_sos_id="<s>"

tgt_eos_id="</s>"

maximum_iterations = tf.round(tf.reduce_max(input_len_vec) * 2)

helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([batch_size], tgt_sos_id), tgt_eos_id)

# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id