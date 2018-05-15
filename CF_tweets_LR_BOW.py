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

# Load target domain test dataset CF labeled turkish tweets
tweets_text, tweets_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_09_05_2018_prccd_pos.txt'), os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_09_05_2018_prccd_neg.txt'))
print("PV  positive samples", (tweets_y.tolist()).count([0, 1]))
print("PV  negative samples", (tweets_y.tolist()).count([1, 0]))

tweets_y = tweets_y[:,1]

# # Plot histogram of text lengths
# text_lengths = [len(x.split()) for x in tweets_text]
# text_lengths = [x for x in text_lengths if x < 50]
# plt.hist(text_lengths, bins=25)
# plt.title('Histogram of # of Words in Texts')

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
vocab = vocab_processor.transform(tweets_text)
embedding_size = len([x for x in vocab_processor.transform(tweets_text)])

# Split up data set into train/test
train_indices = np.random.choice(len(tweets_text), round(len(tweets_text) * 0.7), replace=False)
test_indices = np.array(list(set(range(len(tweets_text))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(tweets_text) if ix in train_indices]
texts_test = [x for ix, x in enumerate(tweets_text) if ix in test_indices]
target_train = [x for ix, x in enumerate(tweets_y) if ix in train_indices]
target_test = [x for ix, x in enumerate(tweets_y) if ix in test_indices]


print("train size", len(texts_train))
print("positive samples", target_train.count([1]))
print("negative samples", target_train.count([0]))

print("test size", len(texts_test))
print("positive samples",  target_test.count([1]))
print("negative samples", target_test.count([0]))
# Setup Index Matrix for one-hot-encoding
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)


x_test_data = tf.placeholder(shape=[sentence_size, len(target_test)], dtype=tf.int32)
y_test_target = tf.placeholder(shape=[1, len(target_test)], dtype=tf.float32)

# Text-Vocab Embedding
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

# Declare model operations
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Prediction operation
prediction = tf.sigmoid(model_output)
_, label_auc = tf.metrics.auc(y_target, prediction)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init =  tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init)

# Start Logistic Regression
print('Starting Training Over {} Sentences.'.format(len(texts_train)))
loss_vec = []
train_acc_all = []
train_acc_avg = []
train_auc_all= []
train_sk_auc_all= []
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    #if (ix + 1) % 10 == 0:
        #print('Training Observation #' + str(ix + 1) + ': Loss = ' + str(temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    auc = sess.run(label_auc, feed_dict={x_data: t, y_target: y_data})
    #sk_auc = metrics.roc_auc_score(y_data,temp_pred)
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    train_auc_all.append(auc)

    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# Get test set accuracy
print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
test_acc_all = []
test_auc_all =[]
test_sk_auc_all =[]
test_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]
    if (ix + 1) % 50 == 0:
        print('Test Observation #' + str(ix + 1))

        # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    auc = sess.run(label_auc, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix] == np.round(temp_pred)
    test_auc_all.append(auc)
    test_acc_all.append(test_acc_temp)
    if len(test_acc_all) >= 50:
        test_acc_avg.append(np.mean(test_acc_all[-50:]))
print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

print('\nOverall Training auc: {}'.format(np.mean(train_auc_all)))
print('\nOverall Test auc: {}'.format(np.mean(test_auc_all)))
print('\n')

# test_vocab = vocab_processor.fit_transform(texts_test)
# test_set = np.array(list(test_vocab))
# print(test_set)
# temp_pred = sess.run(prediction, feed_dict={x_test_data: test_set, y_test_target: np.array([target_test])})
# test_auc_temp = metrics.roc_auc_score(target_test, temp_pred)
#
# print('\nOverall Test auc: {}'.format(np.mean(test_auc_temp)))

# Plot training accuracy over time
# plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
# plt.plot(range(len(test_acc_avg)), test_acc_avg, 'r--', label='test Accuracy')
# plt.title('Avg Training Acc Over Past 50 Generations')
# plt.xlabel('Generation')
# plt.ylabel(' Accuracy')
# plt.show()

