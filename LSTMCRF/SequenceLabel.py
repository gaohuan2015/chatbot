import numpy as np
import tensorflow as tf

# Data settings.
num_examples = 30
num_words = 20
num_features = 200
num_tags = 5

# Random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

with tf.Graph().as_default():
  with tf.Session() as session:
    x_t = tf.constant(x)
    y_t = tf.constant(y)
    equence_lengths_t = tf.constant(sequence_lengths)
    weights = tf.get_variable("weights", [num_features, num_tags])
    matricized_x_t = tf.reshape(x_t, [-1, num_features])
    matricized_unary_scores = tf.matmul(matricized_x_t, weights)
    unary_scores = tf.reshape(matricized_unary_scores, [num_examples, num_words, num_tags])
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, y_t, equence_lengths_t)
    loss = tf.reduce_mean(-log_likelihood)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    session.run(tf.global_variables_initializer())
    for i in range(4000):
        tf_unary_scores, tf_transition_params, _ = session.run(
              [unary_scores, transition_params, train_op])
        if i % 100 == 0:
            correct_labels = 0
            total_labels = 0
            for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, y, sequence_lengths):
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, tf_transition_params)
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Accuracy: %.2f%%" % accuracy)


