import tensorflow as tf
import input_data_pulsar
import numpy as np
p_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309p_pfd_sub_band_int.pkl'
n_path = 'H:\\pg\\1-SKA\\Data\\p309\\pkl_files\\p309n_pfd_sub_band_int.pkl'
all_Candidates = input_data_pulsar.read_data(p_path, n_path,test_part = 0.3)
f_len, t_len, p_len = all_Candidates.train.data_size
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.InteractiveSession(config=config)
sess = tf.InteractiveSession()
x = tf.placeholder("float",shape=(None,f_len,p_len,1))
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
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Dense layer
W_fc1 = weight_variable([int(f_len/4)*int(p_len/4)*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,int(f_len/4)*int(p_len/4)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# adding other attributes
other_attr = tf.placeholder("float",shape=(None,2))
h_fc2 = tf.concat([h_fc1,other_attr],1)

# dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc2,keep_prob)

# output layer
W_fc2 = weight_variable([1026,2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_predictin = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictin,'float'))
sess.run(tf.initialize_all_variables())

def get_fit_cnn_input(input_list):
    for sample_no in range(len(input_list)):
        input_list[sample_no] = input_list[sample_no].reshape(f_len,p_len,1)
    return np.asarray(input_list)
for i in range(2000):
    batch = all_Candidates.train.next_batch(50)
    cnn_input = get_fit_cnn_input(batch[0])
    other_attrs = []
    other_attrs.append(batch[2])
    other_attrs.append(batch[3])
    other_attrs = np.asarray(other_attrs).T
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:cnn_input,y_:batch[4],other_attr:other_attrs, keep_prob:0.5})
        print("step:%d, training_accuracy: %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:cnn_input,y_:batch[4],other_attr:other_attrs, keep_prob:0.5})
