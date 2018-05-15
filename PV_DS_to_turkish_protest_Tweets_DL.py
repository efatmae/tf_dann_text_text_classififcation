
# coding: utf-8

import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import os
from flip_gradient import flip_gradient
from util_pv import *
from text_CNN_feature_extraction import TextCNN
from tensorflow.contrib import learn
from Data_Preprocessing import text_files_preprocessing as tfp
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import time

BASE_DIR = os.path.dirname(__file__)
start_time = time.time()


GT_actual = open(os.path.join(BASE_DIR, "results/GT_actual.txt"), "w")
probabilities= open(os.path.join(BASE_DIR, "results/probabilities.txt"), "w")
predict = open(os.path.join(BASE_DIR, "results/predict.txt"), "w")
test_set = open(os.path.join(BASE_DIR, "results/test_set_vocab.txt"), "w")
training_Set = open(os.path.join(BASE_DIR, "results/training_set_vocab.txt"), "w")
training_labels_Set = open(os.path.join(BASE_DIR, "results/training_set_labels.txt"), "w")



# Process source domain dataset PV dataset
PV_DS_x_text, PV_DS_y = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/PV_DS/PV_Dann_Data/pos_vio_news.txt'), os.path.join(BASE_DIR, 'Data/PV_DS/PV_Dann_Data/neg_vio_news.txt'))
training_Set.write("\n".join(map(lambda x: str(x), PV_DS_x_text)))
training_labels_Set.write("\n".join(map(lambda x: str(x[1]), PV_DS_y)))
print("labeled (source) dataset size",len(PV_DS_x_text))
PV_DS_AVG_document_length = sum([len(x.split(" ")) for x in PV_DS_x_text]) // len(PV_DS_x_text)


# Load target domain test dataset CF labeled turkish tweets
tweets_x_text_test, tweets_y_test = tfp.load_data_and_labels(os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/turkish_protest_test_pos_prccd2.txt'), os.path.join(BASE_DIR, 'Data/Turkish_protests_tweets/turkish_protest_test_neg_prccd2.txt'))
print("CF labeled (target) dataset size (target test set)",len(tweets_x_text_test))


tweets_test_AVG_document_length = sum([len(x.split(" ")) for x in tweets_x_text_test]) // len(tweets_x_text_test)


seq_length = tweets_test_AVG_document_length
print (seq_length)


#source dataset Build vocabulary
vocab_processor_PV = learn.preprocessing.VocabularyProcessor(seq_length)
PV_DS_x = np.array(list(vocab_processor_PV.fit_transform(PV_DS_x_text)))
print("source dataset vocabulary size",len(vocab_processor_PV.vocabulary_))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(PV_DS_y)))
PV_DS_x_shuffled = PV_DS_x[shuffle_indices]
PV_DS_y_shuffled = PV_DS_y[shuffle_indices]



# Split train/test set
dev_sample_index = -1 * int(0.3 * float(len(PV_DS_y)))

print("dev_sample_index", dev_sample_index)
PV_DS_x_train, PV_DS_x_test = PV_DS_x_shuffled[:dev_sample_index], PV_DS_x_shuffled[dev_sample_index:]
PV_DS_y_train, PV_DS_y_test = PV_DS_y_shuffled[:dev_sample_index], PV_DS_y_shuffled[dev_sample_index:]
print("PV training sample size", len(PV_DS_x_train))
print("PV test sample size", len(PV_DS_x_test))


batch_source_ratio = len(PV_DS_x_train)



#target test tweets Build vocabulary
vocab_processor_tweets_test = learn.preprocessing.VocabularyProcessor(seq_length)
test_set.write("\n".join(map(lambda x: str(x), tweets_x_text_test)))
tweets_x_test = np.array(list(vocab_processor_tweets_test.fit_transform(tweets_x_text_test)))


vocab_size = vocab_processor_PV.vocabulary_


# Create a mixed dataset for TSNE visualization
PV_num_test =  1 * int(0.01 * float(len(PV_DS_y_shuffled)))
tweets_num_test = len(tweets_y_test)
print(PV_num_test)



class DANNModel(object):
    """Simple domain adaptation model."""
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

        # CNN model for sentence feature extraction
        with tf.variable_scope('feature_extractor'):
            cnn = TextCNN(self.X, self.y,
                          sequence_length=self.X_length,
                          num_classes=self.y_length,
                          vocab_size=len(vocab_size),
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
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_source_ratio, -1])#here they use only label from the source dataset
            print("no. labeled source features: ", batch_source_ratio)
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_source_ratio, -1]) #here they use only label from the source dataset
            print("no. source features labels: ", batch_source_ratio)
            self.classify_labels = tf.cond(self.train, source_labels, all_labels) #Returns either fn1() or fn2() based on the boolean  pred

            #forward propagation
            W_fc0 = weight_variable([384, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 2])
            b_fc2 = bias_variable([2])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):

            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l) # self.l is the lamda

            d_W_fc0 = weight_variable([384, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits,labels=self.domain)


# In[4]:

# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = DANNModel()

    learning_rate = tf.placeholder(tf.float32, [])

    #TensorFlow provides several operations that you can use to perform common math computations that reduce various dimensions of a tensor.
    pred_loss = tf.reduce_mean(model.pred_loss) #Computes the mean of elements across dimensions of a tensor.
    domain_loss = tf.reduce_mean(model.domain_loss)
    pred_prob = model.pred
    total_loss = pred_loss + domain_loss
    #label_predictions = tf.app.flags.DEFINE_string("outfile","output file for prediction for probabilities")
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # Evaluation
    #here they compare the labels of the data to the predicted labels
    #argmax : Returns the index with the largest value across axes of a tensor.
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1)) #Returns the truth value of (x == y) element-wise.A Tensor of type bool
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
    pred = tf.cast(tf.argmax(model.pred, 1), tf.float32)
    actual = tf.cast(tf.argmax(model.classify_labels, 1), tf.float32)
    label_auc,opt_up = tf.metrics.auc(actual, pred)

    #scalars for tensorboard graph
    tfs = tf.summary.scalar("loss", total_loss)
    # Create a summary to monitor accuracy tensor
    aucs = tf.summary.scalar("accuracy", label_auc)

    # Merge all summaries into a single op
    merged = tf.summary.merge_all()

# In[5]:

# Params
num_steps = 5000

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        gen_source_only_batch = batch_generator(
            [PV_DS_x_train, PV_DS_y_train], 128)

        summary_writer = tf.summary.FileWriter('tensorBorad_logs/log_PV_to_TurkishTweets_networks_29_01_2018', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Training loop
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1 #lamda
            lr = 0.01 / (1. + 10 * p)**0.75 #learning rate

            # Training step
            if training_mode == 'source':
                X, y = next(gen_source_only_batch) #PV_DS_x_train, PV_DS_y_train
                _, batch_loss= sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l,learning_rate: lr})




        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                            feed_dict={model.X: PV_DS_x_test, model.y: PV_DS_y_test,
                                       model.train: False})

        _, source_auc = sess.run([label_auc,opt_up],
                            feed_dict={model.X: PV_DS_x_test, model.y: PV_DS_y_test,
                                       model.train: False})
        src_pred2 = sess.run(pred,
                            feed_dict={model.X: PV_DS_x_test, model.y: PV_DS_y_test,
                                       model.train: False})

        src_actual2 = sess.run(actual,
                            feed_dict={model.X: PV_DS_x_test, model.y: PV_DS_y_test,
                                       model.train: False})

        sess.run(tf.local_variables_initializer())

        target_acc = sess.run(label_acc,
                            feed_dict={model.X: tweets_x_test, model.y: tweets_y_test,
                                       model.train: False})
        _, target_auc = sess.run([label_auc,opt_up],
                            feed_dict={model.X: tweets_x_test, model.y: tweets_y_test,
                                       model.train: False})

        pred_prob2 = sess.run(pred_prob,
                            feed_dict={model.X: tweets_x_test, model.y: tweets_y_test,
                                       model.train: False})

        pred2 = sess.run(pred,
                            feed_dict={model.X: tweets_x_test, model.y: tweets_y_test,
                                       model.train: False})

        actual2 = sess.run(actual,
                            feed_dict={model.X: tweets_x_test, model.y: tweets_y_test,
                                       model.train: False})


    return source_acc, source_auc, src_pred2, src_actual2, target_acc, target_auc, pred_prob2, pred2, actual2



print('\nSource only training')
source_acc, source_auc, src_pred3, src_actual3, target_acc, target_auc, pred_prob3, pred3, actual3 = train_and_evaluate('source', graph, model)
print('Source (PV dataset) accuracy:', source_acc)
print('Source (PV dataset) auc:', source_auc)
print('Source (PV dataset) confustion matrix:', metrics.confusion_matrix(src_actual3, src_pred3))
print('Target (Turkish protest Tweets) accuracy:', target_acc)
print('Target (Turkish protest Tweets) auc:', target_auc)
print('Target (Turkish protest Tweets) sk_accuracy:', metrics.accuracy_score(actual3, pred3))
print('Target (Turkish protest Tweets) SK_auc:', metrics.roc_auc_score(actual3, pred3))
print('Target (Turkish protest Tweets) confustion matrix:', metrics.confusion_matrix(actual3, pred3))

GT_actual.write("\n".join(map(lambda x: str(x), list(actual3))))
predict.write("\n".join(map(lambda x: str(x), list(pred3))))
probabilities.write("\n".join(map(lambda x: str(x), list(pred_prob3))))


print("--- %s seconds ---" % (time.time() - start_time))
