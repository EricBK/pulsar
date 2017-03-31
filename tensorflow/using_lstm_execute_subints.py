import tensorflow as tf
import input_data_pulsar
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
p_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309p_pfd_sub_band_int.pkl'
n_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309n_pfd_sub_band_int.pkl'
all_Candidates = input_data_pulsar.read_data(p_path, n_path,test_part = 0.3)
f_len, t_len, p_len = all_Candidates.train.data_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 3层 GRU
# hyerparameters
ALPHASIZE = 64
CELLSIZE = p_len
NLAYERS = 3
SEQLEN = 18
BATCHSIZE = 50

learning_rate = 0.001
dropout_keep = 1.0
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# input size
X = tf.placeholder(tf.float32,[None, t_len, p_len]) # [Batchsize,t_len, f_len]
Y_ = tf.placeholder(tf.float32, [None,2]) # [batchsize, class]
#X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)
#Yd_ = tf.placeholder(tf.uint8, [None, None])
#Y_ = tf.one_hot(Yd_, ALPHASIZE, 1.0, 0.0)
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])


cell = rnn.GRUCell(CELLSIZE)
mcell = rnn.MultiRNNCell([cell]*NLAYERS, state_is_tuple = False)
Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state = Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence
# softmax output layer
#Hf = tf.reshape(Hr, [-1,CELLSIZE*p_len])

#YLogits = layers.linear(Hf, ALPHASIZE)
#Y = tf.nn.softmax(YLogits)
lstm_W_fc1 = weight_variable([CELLSIZE*p_len,1024])
lstm_b_fc1 = bias_variable([1024])
lstm_h_pool2_flat = tf.reshape(Hr, [-1,CELLSIZE * p_len])
lstm_h_fc1 = tf.nn.relu(tf.matmul(lstm_h_pool2_flat,lstm_W_fc1) + lstm_b_fc1)

# output layer
W_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(lstm_h_fc1, W_fc2)+b_fc2)

# loss and training step
loss = -tf.reduce_sum(Y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)



correct_predictin = tf.equal(tf.argmax(y_conv,1),tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictin,'float'))
istate = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

for i in range(1000):
#    istate = np.zeros([batchsize, CELLSIZE * NLAYERS])  # initial zero input state

    batch = all_Candidates.train.next_batch(BATCHSIZE)
    lstm_input = batch[1]
    dic = {X:lstm_input,Y_:batch[4],Hin:istate}
    if i%10 == 0:
        #train_accuracy = accuracy.eval(feed_dict=dic)
        _, acc, y, outH = sess.run([train_step, accuracy, y_conv, H, ], feed_dict=dic)
        print("step:%d, training_accuracy: %g"%(i,acc))
    _, acc, y , outH = sess.run([train_step, accuracy,y_conv, H,], feed_dict = dic)
    istate = outH
    #train_step.run(feed_dict=dic)
