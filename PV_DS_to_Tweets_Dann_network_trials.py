
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import os
from flip_gradient import flip_gradient
from util_pv import *
from text_CNN_feature_extraction import TextCNN
from tensorflow.contrib import learn
from Data_Preprocessing import text_files_preprocessing as tfp
import time

BASE_DIR = os.path.dirname(__file__)
start_time = time.time()

# Process source domain dataset rp-polaritydata

PV_DS_x_text, PV_DS_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/PV_DS/PV_Dann_Data/pos_vio_news.txt'), os.path.join(BASE_DIR, 'Data/PV_DS/PV_Dann_Data/neg_vio_news.txt'))
PV_DS_x_text = PV_DS_x_text[:-1]
PV_DS_y = PV_DS_y[:-1]
print(len(PV_DS_x_text))
PV_DS_AVG_document_length = sum([len(x.split(" ")) for x in PV_DS_x_text]) // len(PV_DS_x_text)

# Load target domain dataset IMDB
tweets_x_text, tweets_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/sentiment_tweets/pos_tweets_prccd.txt'), os.path.join(BASE_DIR, 'Data/sentiment_tweets/neg_tweets_prccd.txt'))
#tweets_x_text, tweets_y = tfp.load_data_and_labels(os.path.join(BASE_DIR,'Data/IMDB/pos_movies.txt'), os.path.join(BASE_DIR,'Data/IMDB/neg_movies.txt'))
print(len(tweets_x_text))
tweets_AVG_document_length = sum([len(x.split(" ")) for x in tweets_x_text]) // len(tweets_x_text)
print(tweets_AVG_document_length)
PV_DS_AVG_document_length = sum([len(x.split(" ")) for x in PV_DS_x_text]) // len(PV_DS_x_text)
print(PV_DS_AVG_document_length)


seq_length = (PV_DS_AVG_document_length + tweets_AVG_document_length) // 2

print(seq_length)
# Build vocabulary
vocab_processor_PV = learn.preprocessing.VocabularyProcessor(seq_length)
PV_DS_x = np.array(list(vocab_processor_PV.fit_transform(PV_DS_x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(PV_DS_y)))
PV_DS_x_shuffled = PV_DS_x[shuffle_indices]
PV_DS_y_shuffled = PV_DS_y[shuffle_indices]

# Build vocabulary
vocab_processor_tweets = learn.preprocessing.VocabularyProcessor(seq_length)
tweets_x = np.array(list(vocab_processor_tweets.fit_transform(tweets_x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(tweets_y)))
tweets_x_shuffled = tweets_x[shuffle_indices]
tweet_y_shuffled = tweets_y[shuffle_indices]

# Split train/test set

dev_sample_index = -1 * int(0.1 * float(len(PV_DS_y)))

PV_DS_x_train, PV_DS_x_test = PV_DS_x_shuffled[:dev_sample_index], PV_DS_x_shuffled[dev_sample_index:]
PV_DS_y_train, PV_DS_y_test = PV_DS_y_shuffled[:dev_sample_index], PV_DS_y_shuffled[dev_sample_index:]

dev_sample_index = -1 * int(0.1 * float(len(tweets_y)))

tweet_x_train, tweet_x_test = tweets_x_shuffled[:dev_sample_index], tweets_x_shuffled[dev_sample_index:]
tweet_y_train, tweet_y_test = tweet_y_shuffled[:dev_sample_index], tweet_y_shuffled[dev_sample_index:]

# Create a mixed dataset for TSNE visualization
num_test = 500
combined_test_txt = np.vstack([PV_DS_x_test[:num_test], tweet_x_test[:num_test]])
combined_test_labels = np.vstack([PV_DS_y_test[:num_test], tweet_y_test[:num_test]]) # they use hte labels of hte source domian dataset with the target domain dataset
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]), #domain labels 1st half 0 source and 2nd hald target 1
        np.tile([0., 1.], [num_test, 1])])

batch_size = 128

class MNISTModel(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X_length=PV_DS_x_train.shape[1]
        self.y_length =PV_DS_y_train.shape[1]
        self.X = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        self.num_filters = 128
        self.filter_sizes = [3,4,5]

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            cnn = TextCNN(self.X, self.y,
                          sequence_length=self.X_length,
                          num_classes=self.y_length,
                          vocab_size=len(vocab_processor_PV.vocabulary_),
                          embedding_size=128,
                          filter_sizes=self.filter_sizes,
                          num_filters=self.num_filters)

            # The domain-invariant feature
            self.feature = cnn.h_pool_flat
            self.h_drop = cnn.h_drop

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):

            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.h_drop
            source_features = lambda: tf.slice(self.h_drop, [0, 0], [batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels) #Return either fn1() or fn2() based on the boolean predicate pred

            #forward propagation
            W_fc0 = weight_variable([384, 2])
            b_fc0 = bias_variable([2])

            self.scores = tf.nn.xw_plus_b(classify_feats, W_fc0, b_fc0, name="scores")
            logits = self.scores
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):

            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.h_drop, self.l) # self.l is the lamda

            d_W_fc0 = weight_variable([384, 2])
            d_b_fc0 = bias_variable([2])

            self.d_scores = tf.nn.xw_plus_b(feat, d_W_fc0, d_b_fc0, name="d_scores")
            d_logits = self.d_scores
            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits,labels=self.domain)


# In[4]:

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MNISTModel()

    learning_rate = tf.placeholder(tf.float32, [])

    #TensorFlow provides several operations that you can use to perform common math computations that reduce various dimensions of a tensor.
    pred_loss = tf.reduce_mean(model.pred_loss) #Computes the mean of elements across dimensions of a tensor.
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)#Optimizer that implements the Momentum algorithm
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    #TensorFlow provides several operations that you can use to add comparison operators to your graph.
    #here they compare the labels of the data to the predicted labels
    #argmax : Returns the index with the largest value across axes of a tensor.
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1)) #Returns the truth value of (x == y) element-wise.A Tensor of type bool
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))


# In[5]:

# Params
num_steps = 8000
print(num_steps)
def train_and_evaluate(training_mode, graph, model, verbose=False):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        #summary_writer = tf_basic_trials.summary.FileWriter('tensorBorad_logs/log_IMDB_To_Amazon_movies_DANN_networks', sess.graph)
        sess.run(tf.global_variables_initializer())

        # Batch generators
        #according to the paper the source domain sample with the labels know and the target domain sampels with the labels unkown
        gen_source_batch = batch_generator(
            [PV_DS_x_train, PV_DS_y_train], batch_size // 2)
        gen_target_batch = batch_generator(
            [tweet_x_train, tweet_y_train], batch_size // 2) #here they use hte labels of hte source training along wiht samples fro mthe target . instead of the target labels
        gen_source_only_batch = batch_generator(
            [PV_DS_x_train, PV_DS_y_train], batch_size)
        gen_target_only_batch = batch_generator(
            [tweet_x_train, tweet_y_train], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                   np.tile([0., 1.], [batch_size // 2, 1])])

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1 #lamda
            lr = 0.01 / (1. + 10 * p)**0.75 #learning rate

            # Training step
            if training_mode == 'dann':

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])


                _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                             feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                        model.train: True, model.l: l, learning_rate: lr})

                if verbose and i % 100 == 0:
                    print('loss: %f  d_acc: %f  p_acc: %f  p: %f  l: %f  lr: %f' % (batch_loss, d_acc, p_acc, p, l, lr))

            elif training_mode == 'source':
                X, y = next(gen_source_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l,learning_rate: lr})

            elif training_mode == 'target':
                X, y = next(gen_target_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: lr})

        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                            feed_dict={model.X: PV_DS_x_test, model.y: PV_DS_y_test,
                                       model.train: False})

        target_acc = sess.run(label_acc,
                            feed_dict={model.X: tweet_x_test, model.y: tweet_y_test,
                                       model.train: False})

        test_domain_acc = sess.run(domain_acc,
                            feed_dict={model.X: combined_test_txt,
                                       model.domain: combined_test_domain, model.l: 1.0})

        test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_txt})

    return source_acc, target_acc, test_domain_acc, test_emb


print('\nSource only training')
source_acc, target_acc, d_acc, source_only_emb = train_and_evaluate('source', graph, model)
print('Source (Amazon movies reviews) accuracy:', source_acc)
print('Target (Tweets sentiment) accuracy:', target_acc)
print('Domain accuracy:', d_acc)

print('\nDomain adaptation training')
source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model)
print('Source (Amazon movies reviews) accuracy:', source_acc)
print('Target (Tweets sentiment) accuracy:', target_acc)
print('Domain accuracy:', d_acc)

print('\nTarget only training')
source_acc, target_acc, d_acc, target_emb = train_and_evaluate('target', graph, model)
print('Source (Amazon movies reviews) accuracy:', source_acc)
print('Target (Tweets sentiment) accuracy:', target_acc)
print('Domain accuracy:', d_acc)

# In[6]:

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
source_only_tsne = tsne.fit_transform(source_only_emb)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
target_only_tsne = tsne.fit_transform(target_emb)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
dann_tsne = tsne.fit_transform(dann_emb)



plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), BASE_DIR + '/Graphs/PV_to_tweets_dann/', 'Source only')
plot_embedding(target_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), BASE_DIR + '/Graphs/PV_to_tweets_dann/', 'Target only')
plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), BASE_DIR + '/Graphs/PV_to_tweets_dann/', 'Domain Adaptation')

print("--- %s seconds ---" % (time.time() - start_time))
