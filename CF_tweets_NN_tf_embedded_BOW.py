import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
from Data_Preprocessing import text_files_preprocessing as tfp
from sklearn import metrics

BASE_DIR = os.path.dirname(__file__)
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Choose max text word length at 25
sentence_size = 10
min_word_freq = 3
Words_Features = 'words'
Embedding_size = 50
n_words = 0

def bag_of_words_model(features, target):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(target, 2, 1, 0)
  features = tf.contrib.layers.bow_encoder(
      features, vocab_size=n_words, embed_dim=Embedding_size, scope="input_layer")
  hidden_layer1 = tf.contrib.layers.fully_connected(features, 100, scope="hidden_layer1")
  logits = tf.contrib.layers.fully_connected(hidden_layer1, 2, scope="output_layer",
      activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)
  return (
      {'class': tf.argmax(logits, 1),
       'prob': tf.nn.softmax(logits)},
      loss, train_op)

# Load target domain test dataset CF labeled turkish tweets
tweets_text, tweets_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_09_05_2018_prccd_pos.txt'), os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_09_05_2018_prccd_neg.txt'))
print("PV  positive samples", (tweets_y.tolist()).count([0, 1]))
print("PV  negative samples", (tweets_y.tolist()).count([1, 0]))

tweets_y = tweets_y[:,1]
# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size , min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
vocab_processor.transform(tweets_text)
embedding_size = len([x for x in vocab_processor.transform(tweets_text)])
print("embedding size = ", embedding_size)
# Split up data set into train/test
train_indices = np.random.choice(len(tweets_text), round(len(tweets_text) * 0.7), replace=False)
test_indices = np.array(list(set(range(len(tweets_text))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(tweets_text) if ix in train_indices]
texts_test = [x for ix, x in enumerate(tweets_text) if ix in test_indices]
target_train = [x for ix, x in enumerate(tweets_y) if ix in train_indices]
target_test = [x for ix, x in enumerate(tweets_y) if ix in test_indices]

# Process vocabulary
texts_train = np.array(list(vocab_processor.fit_transform(texts_train)))
texts_test = np.array(list(vocab_processor.transform(texts_test)))
n_words = len(vocab_processor.vocabulary_)

print('Total words: %d' % n_words)

print("train size", len(texts_train))
print("positive samples", target_train.count([1]))
print("negative samples", target_train.count([0]))

print("test size", len(texts_test))
print("positive samples",  target_test.count([1]))
print("negative samples", target_test.count([0]))

n_words = len(vocab_processor.vocabulary_)
print('no. words', n_words)

classifier = learn.Estimator(model_fn=bag_of_words_model)
# Train and predict
classifier.fit(texts_train, target_train, steps=10000)
y_predicted = [ p['class'] for p in classifier.predict(texts_test, as_iterable=True)]
score = metrics.accuracy_score(target_test, y_predicted)
auc = metrics.roc_auc_score (target_test, y_predicted)
print('Accuracy: {0:f}'.format(score))
print('auc: {0:f}'.format(auc))