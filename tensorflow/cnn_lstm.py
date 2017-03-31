import tensorflow as tf
import input_data_pulsar
import numpy as np
from tensorflow.contrib import rnn
p_path = 'E:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309p_pfd_sub_band_int.pkl'
n_path = 'E:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309n_pfd_sub_band_int.pkl'
all_Candidates = input_data_pulsar.read_data(p_path, n_path,test_part = 0.3)
f_len, t_len, p_len = all_Candidates.train.data_size
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.InteractiveSession(config=config)
sess = tf.InteractiveSession()
X_subbands = tf.placeholder("float",shape=(None,f_len,p_len,1))
y_ = tf.placeholder("float",shape=(None,2))
# 首先构建一个cnn模型
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积和池化
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')

# 第一层卷积
cnn_W_conv1 = weight_variable([5,5,1,32])
cnn_b_conv1 = bias_variable([32])

cnn_h_conv1 = tf.nn.relu(conv2d(X_subbands,cnn_W_conv1) + cnn_b_conv1)
cnn_h_pool1 = max_pool_2x2(cnn_h_conv1)

# 第二层卷积
cnn_W_conv2 = weight_variable([5,5,32,64])
cnn_b_conv2 = bias_variable([64])

cnn_h_conv2 = tf.nn.relu(conv2d(cnn_h_pool1,cnn_W_conv2) + cnn_b_conv2)
cnn_h_pool2 = max_pool_2x2(cnn_h_conv2)

# Dense layer
cnn_W_fc1 = weight_variable([int(f_len/4)*int(p_len/4)*64,1024])
cnn_b_fc1 = bias_variable([1024])

cnn_h_pool2_flat = tf.reshape(cnn_h_pool2, [-1,int(f_len/4)*int(p_len/4)*64])
cnn_h_fc1 = tf.nn.relu(tf.matmul(cnn_h_pool2_flat,cnn_W_fc1) + cnn_b_fc1)
# adding other attributes
cnn_other_attr = tf.placeholder("float",shape=(None,2))
cnn_h_fc2 = tf.concat([cnn_h_fc1,cnn_other_attr],1)

# dropout
cnn_keep_prob = tf.placeholder('float')
cnn_h_fc1_drop = tf.nn.dropout(cnn_h_fc2,cnn_keep_prob)


# LSTM model
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
X_subints = tf.placeholder(tf.float32,[None, t_len, p_len]) # [Batchsize,t_len, f_len]
#Y_ = tf.placeholder(tf.float32, [None,2]) # [batchsize, class]
#X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)
#Yd_ = tf.placeholder(tf.uint8, [None, None])
#Y_ = tf.one_hot(Yd_, ALPHASIZE, 1.0, 0.0)
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])


cell = rnn.GRUCell(CELLSIZE)
mcell = rnn.MultiRNNCell([cell]*NLAYERS, state_is_tuple = False)
Hr, H = tf.nn.dynamic_rnn(mcell, X_subints, initial_state = Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

lstm_W_fc1 = weight_variable([CELLSIZE*p_len,1024])
lstm_b_fc1 = bias_variable([1024])
lstm_h_pool2_flat = tf.reshape(Hr, [-1,CELLSIZE * p_len])
lstm_h_fc1 = tf.nn.relu(tf.matmul(lstm_h_pool2_flat,lstm_W_fc1) + lstm_b_fc1)


final_feature = tf.concat([cnn_h_fc1_drop,lstm_h_fc1],1)

# output layer
W_fc2 = weight_variable([1026+1024,2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(final_feature, W_fc2)+b_fc2)

# loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_predictin = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictin,'float'))
istate = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])  # initial zero input state
sess.run(tf.initialize_all_variables())

def get_fit_cnn_input(input_list):
    for sample_no in range(len(input_list)):
        input_list[sample_no] = input_list[sample_no].reshape(f_len,p_len,1)
    return np.asarray(input_list)
for i in range(2000):
    batch = all_Candidates.train.next_batch(BATCHSIZE)
    cnn_input = get_fit_cnn_input(batch[0])
    other_attrs = []
    other_attrs.append(batch[2])
    other_attrs.append(batch[3])
    other_attrs = np.asarray(other_attrs).T
    feed_dict = {X_subbands:cnn_input,X_subints:batch[1],Hin:istate,y_:batch[4],cnn_other_attr:other_attrs, cnn_keep_prob:0.5}
    if i%10 == 0:
        _, acc, y, outH = sess.run([train_step, accuracy, y_conv, H, ], feed_dict=feed_dict)
        print("step:%d, training_accuracy: %g"%(i,acc))
    _, acc, y, outH = sess.run([train_step, accuracy, y_conv, H, ], feed_dict=feed_dict)
    #train_step.run(session = sess,feed_dict=feed_dict)
    istate = outH
