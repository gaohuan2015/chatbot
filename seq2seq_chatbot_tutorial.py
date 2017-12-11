import re
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *

enc_sentence_length = 10
dec_sentece_length = 10
batche_size = 4

input_batches = [
    ['Hi What is your name?', 'Nice to meet you!'],
    ['Which programming language do you use?', 'See you later.'],
    ['Where do you live?', 'What is your major?'],
    ['What do you want to drink?', 'What is your favorite beer?'],
    ['will you tell me something about yourself?', 'what do you think are your strengths and weaknesses?']]

target_batches = [
    ['Hi this is Jaemin.', 'Nice to meet you too!'],
    ['I like Python.', 'Bye Bye.'],
    ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
    ['Beer please!', 'Leffe brown!'],
    ['my major is Computer science and technology', 'I have positive opinions to work and life']]

all_input_sentence = []
for input_batch in input_batches:
    all_input_sentence.extend(input_batch)

all_target_sentence = []
for target_batch in target_batches:
    all_target_sentence.extend(target_batch)


def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens


def build_vocab(sentences, is_target=False, max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()
    for sentence in sentences:
        token = tokenizer(sentence)
        word_counter.update(token)
    if max_vocab_size is None:
        max_vocab_size = len(word_counter)
    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_idx = 2
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab['_PAD'] = 1
        vocab_idx = 1
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    for key, value in vocab.items():
        reverse_vocab[value] = key
    return vocab, reverse_vocab, max_vocab_size


enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(all_input_sentence)
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(all_target_sentence, is_target=True)


def token2idx(word, vocab):
    return vocab[word]


def sent2idx(sent, vocab=enc_vocab, max_sentence=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [1] * pad_length, current_length


def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]


def idx2sent(indices, reverse_vocab=dec_reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])


n_epoch = 2000
n_enc_layer = 3
n_dec_layer = 3
hidden_size = 30
enc_emb_size = 30
dec_emb_size = 30

tf.reset_default_graph()
enc_inputs = tf.placeholder(tf.int32, shape=[None, enc_sentence_length])
sequence_lengths = tf.placeholder(tf.int32, shape=[None])
dec_inputs = tf.placeholder(tf.int32, shape=[None, dec_sentece_length + 1])
enc_inputs_t = tf.transpose(enc_inputs, [1, 0])
dec_inputs_t = tf.transpose(dec_inputs, [1, 0])

with tf.variable_scope('encoder'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    enc_cell = EmbeddingWrapper(enc_cell, enc_vocab_size + 1, enc_emb_size)
    enc_outputs, enc_last_state = tf.contrib.rnn.static_rnn(
        cell=enc_cell,
        inputs=tf.unstack(enc_inputs_t),
        sequence_length=sequence_lengths,
        dtype=tf.float32)

dec_outputs = []
dec_predictions = []

with tf.variable_scope("decoder") as scope:
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    dec_cell = EmbeddingWrapper(dec_cell, dec_vocab_size + 2, dec_emb_size)
    dec_cell = OutputProjectionWrapper(dec_cell, dec_vocab_size + 2)
    for i in range(dec_sentece_length + 1):
        if i == 0:
            input_ = dec_inputs_t[i]
            state = enc_last_state
        else:
            scope.reuse_variables()
            input_ = dec_prediction
        dec_output, state = dec_cell(input_, state)
        dec_prediction = tf.argmax(dec_output, axis=1)
        dec_outputs.append(dec_output)
        dec_predictions.append(dec_prediction)

predictions = tf.transpose(tf.stack(dec_predictions), [1, 0])
labels = tf.one_hot(dec_inputs_t, dec_vocab_size + 2)
logits = tf.stack(dec_outputs)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_history = []
    for epoch in range(n_epoch):
        all_preds = []
        epoch_loss = 0
        for input_batch, target_batch in zip(input_batches, target_batches):
            input_token_indices = []
            target_token_indices = []
            sentence_lengths = []

            for input_sent in input_batch:
                input_sent, sent_len = sent2idx(input_sent)
                input_token_indices.append(input_sent)
                sentence_lengths.append(sent_len)

            for target_sent in target_batch:
                target_token_indices.append(
                    sent2idx(target_sent, vocab=dec_vocab, max_sentence=dec_sentece_length, is_target=True))
            batch_preds, batch_loss, _ = sess.run([predictions, loss, training_op],
                                                  feed_dict={enc_inputs: input_token_indices,
                                                             sequence_lengths: sentence_lengths,
                                                             dec_inputs: target_token_indices})
            loss_history.append(batch_loss)
            epoch_loss += batch_loss
            all_preds.append(batch_preds)
            if epoch % 400 == 0:
                print('Epoch', epoch)
                for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                    for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                        print('\t', input_sent)
                        print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                        print('\tCorrent answer:', target_sent)
                print('\tepoch loss: {:.2f}\n'.format(epoch_loss))
