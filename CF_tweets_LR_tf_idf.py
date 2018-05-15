import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.framework import ops
from Data_Preprocessing import text_files_preprocessing as tfp
from sklearn import metrics

BASE_DIR = os.path.dirname(__file__)
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

batch_size = 200
max_features = 1000

# Load target domain test dataset CF labeled turkish tweets
tweets_text, tweets_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_13_04_2018_prccd_pos.txt'), os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/Turkish_tweets_CF_results_13_04_2018_prccd_neg.txt'))
print("PV  positive samples", (tweets_y.tolist()).count([0, 1]))
print("PV  negative samples", (tweets_y.tolist()).count([1, 0]))

tweets_y = tweets_y[:,1]

# Define tokenizer
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words


# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(tweets_text)

# Split up data set into train/test
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8 * sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]

texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(tweets_y) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(tweets_y) if ix in test_indices])

print("train size", texts_train.shape)
print("positive samples",  np.count_nonzero(target_train == 1))
print("negative samples", np.count_nonzero(target_train == 0))

print("test size", texts_test.shape)
print("positive samples",  np.count_nonzero(target_test == 1))
print("negative samples", np.count_nonzero(target_test == 0))

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Initialize placeholders
x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare logistic model (sigmoid in loss function)
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output)) #round the orbabilities to 1 or 0
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct) #(tp+tn) / total
_, label_auc = tf.metrics.auc(y_target, prediction)
# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

# Intitialize Variables
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

sess.run(init)

# Start Logistic Regression
train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_auc = []
test_auc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)

        train_auc_temp = sess.run(label_auc, feed_dict={x_data: rand_x, y_target: rand_y})
        train_auc.append(train_auc_temp)

        test_auc_temp = sess.run(label_auc,
                                 feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_auc.append(test_auc_temp)

    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp, train_auc_temp, test_auc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f}). Train Auc (Test Auc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

accuracy = sess.run(accuracy, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
test_auc = sess.run(label_auc, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})

#test_auc = metrics.roc_auc_score(target_test,pred)
#print("test auc", test_auc)
print("accuracy", accuracy)
print("test auc", test_auc)
# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
