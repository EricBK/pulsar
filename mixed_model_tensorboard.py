
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math
import csv
from models.PMPS.model_config import *
import random
from models.PMPS.data_util.read_attrs import *
import matplotlib.pyplot as plt
import numbers
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

f_len = CandConfig.FRE_LEN
t_len = CandConfig.TIME_LEN
p_len = CandConfig.PHASE_LEN
pkl_dir = 'E:\\pg\\1-SKA\\Data\\PMPS\\PMPS_labeled\\'
all_pkl_filepath = allCandidates(pkl_dir)
train_pkl_file_list = all_pkl_filepath.Good_pkl.copy()
train_pkl_file_list.extend(all_pkl_filepath.RFI_pkl)
print('Good_pkl list',all_pkl_filepath.Good_pkl)
# train_batched: {'attrs':value,'label':value}
print('num of RFI pkl files:%g, Good pkl file num: %g'%(len(all_pkl_filepath.RFI_pkl),len(all_pkl_filepath.Good_pkl)))
train_RFI_pkl_file = all_pkl_filepath.RFI_pkl[int(random.uniform(0,len(all_pkl_filepath.RFI_pkl)))]
train_Good_pkl_file = all_pkl_filepath.Good_pkl[int(random.uniform(0,len(all_pkl_filepath.Good_pkl)))]
print('reading training data...')
print([train_RFI_pkl_file, train_Good_pkl_file])
train_batches = BatchGenerator([train_RFI_pkl_file, train_Good_pkl_file],label_list = [0,1],batch_size=BATCHSIZE)
print('Training samples:%g'%(train_batches.cand_num))

print('f_len:',f_len,'t_len:',t_len,'p_len:',p_len)
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.InteractiveSession(config=config)
sess = tf.InteractiveSession()
with tf.name_scope('inputs'):
    with tf.name_scope('CNN_inputs'):
        X_subbands = tf.placeholder("float",shape=(None,f_len,p_len,1),name='X_subbands') # [batchsize,f_len,p_len] , [None,96,64]
        tf.summary.image('cnn_input',X_subbands)
    with tf.name_scope('RNN_inputs'):
        X_subints = tf.placeholder(tf.float32, [None, t_len, p_len],name='X_subints')  # [batchsize,t_len, f_len]   , [None,256,64]
        tf.summary.image('RNN_input',tf.reshape(X_subints,shape=[-1,t_len,p_len,1]))
    with tf.name_scope('True_label'):
        y_ = tf.placeholder("float",shape=(None,2),name='y_')

def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
# drop_out selu
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

def variable_summaries(var,var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name+'summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

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

cnn_keep_prob = tf.placeholder('float')

def add_conv_layer(inputs, kernal_length, kernal_width, kernal_depth, kernal_num, n_layer, activation_func):
    layer_name = 'conv_layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            cnn_W_conv = weight_variable([kernal_length, kernal_width, kernal_depth , kernal_num])
        with tf.name_scope('bias'):
            cnn_b_conv = bias_variable([kernal_num])
        # X_subbands.shape = [None,96,64]
        with tf.name_scope('Wx_plux_bias'):
            if activation_func == 'SELU':
                cnn_h_conv = selu(conv2d(inputs, cnn_W_conv) + cnn_b_conv)
                cnn_h_conv = dropout_selu(cnn_h_conv, rate=selu_dp_rate)
            elif activation_func == 'RELU':
                cnn_h_conv = tf.nn.relu(conv2d(inputs, cnn_W_conv) + cnn_b_conv)
        with tf.name_scope('max_pool'):
            cnn_h_pool = max_pool_2x2(cnn_h_conv)  # [None, 48, 32]
    return cnn_h_pool
def add_dense_layer(inputs,n_layer, activation_func,cnn_keep_prob, selu_dp_rate,nodes_f,nodes_b):
    layer_name = 'dense_layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            cnn_W_fc = weight_variable([ nodes_f, nodes_b])  # [12*8, 128]
        with tf.name_scope('bias'):
            cnn_b_fc = bias_variable([nodes_b])
        with tf.name_scope('Wx_plus_b'):
            if activation_func == 'SELU':
                cnn_h_fc = selu(tf.matmul(inputs,cnn_W_fc) + cnn_b_fc)
                cnn_h_fc_drop = dropout_selu(cnn_h_fc, rate=selu_dp_rate)
            elif activation_func == 'RELU':
                cnn_h_fc = tf.nn.relu(tf.matmul(inputs,cnn_W_fc) + cnn_b_fc)       # [None,96] * [96,128] --> [None, 128]
                # dropout
                cnn_h_fc_drop = tf.nn.dropout(cnn_h_fc,cnn_keep_prob)
        return cnn_h_fc_drop

def cnn_model():

    with tf.name_scope('CNN_model'):
        kernal_width = 3
        kernal_length = 3
        # 第一层卷积
        cnn_h_output1 = add_conv_layer(inputs=X_subbands, kernal_length = kernal_length, kernal_width = kernal_width,kernal_depth =1,
                                kernal_num = 8, n_layer = 1,activation_func = activation_func)  # shape: (?, 48, 32, 8)
        tf.summary.image('cnn_h_output1',tf.reshape(cnn_h_output1[:,:,:,0],shape=[-1,48,32,1]))
        # 第二层卷积
        cnn_h_output2 = add_conv_layer(inputs=cnn_h_output1,kernal_length=kernal_length, kernal_width=kernal_width, kernal_depth=8,
                                kernal_num = 32, n_layer = 2,activation_func=activation_func) # [None, 24,16]
        tf.summary.image('cnn_h_output2', tf.reshape(cnn_h_output2[:,:,:,0],shape=[-1,24,16,1]))
        # 第三层卷积
        cnn_h_output3 = add_conv_layer(inputs=cnn_h_output2, kernal_length=kernal_length, kernal_width=kernal_width,kernal_depth=32,
                                  kernal_num=64,n_layer=3,activation_func=activation_func) # [None, 12,8]
        tf.summary.image('cnn_h_output3', tf.reshape(cnn_h_output3[:,:,:,0],shape=[-1,12,8,1]))
        convolution_layer = 3

        de_multi = int(math.pow(2,convolution_layer))
        with tf.name_scope('flatten'):
            cnn_h_flat = tf.reshape(cnn_h_output3, [-1,int(f_len/de_multi)*int(p_len/de_multi)*64]) # [None, 96]
        # Dense layer_1
        cnn_fc1_output = add_dense_layer(inputs=cnn_h_flat, n_layer = 1,activation_func=activation_func, cnn_keep_prob=cnn_keep_prob,
                                         selu_dp_rate=selu_dp_rate,nodes_f=int(f_len / de_multi) * int(p_len / de_multi) *64, nodes_b=128)
        # Dense layer_2
        cnn_fc2_output = add_dense_layer(inputs=cnn_fc1_output, n_layer =2,activation_func=activation_func,cnn_keep_prob=cnn_keep_prob,
                                         selu_dp_rate=selu_dp_rate,nodes_f=128, nodes_b=256)

        #tf.summary.histogram('cnn_fc2_output', cnn_fc2_output)
        return cnn_fc2_output

Hin = tf.placeholder(tf.float32, [None, NLAYERS*CELLSIZE ])
def NAS_cell():
    cell = tf.contrib.rnn.NASCell(CELLSIZE, reuse=tf.get_variable_scope().reuse)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
def GRUCell(CELLSIZE):
    with tf.name_scope('GRUCell'):
        return rnn.GRUCell(CELLSIZE)
def LSTM_model():

    with tf.name_scope('RNN_model'):
        dropout_keep = 1.0
        lr = tf.placeholder(tf.float32, name='lr')  # learning rate
        pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
        batchsize = tf.placeholder(tf.int32, name='batchsize')

        #with tf.variable_scope('lSTM'):
        #cell = rnn.GRUCell(CELLSIZE)

        # dropcell = rnn.DropoutWrapper(cell, input_keep_prob=pkeep)
        # multicell = rnn.MultiRNNCell([dropcell for _ in range(NLAYERS)], state_is_tuple=False)
        # multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
        # multicell = rnn.MultiRNNCell([cell]*NLAYERS, state_is_tuple = False)
        cell_type = CELL_TYPE
        if cell_type == 'NAC':
            multicell = tf.contrib.rnn.MultiRNNCell([NAS_cell() for _ in range(NLAYERS)], state_is_tuple=True)
        #    multicell = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(CELLSIZE) for _ in range(NLAYERS)], state_is_tuple=True)
            init_state = tf.placeholder(tf.float32, [NLAYERS, 2, BATCHSIZE, CELLSIZE])
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple([rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(NLAYERS)])
            Hr, H = tf.nn.dynamic_rnn(multicell, X_subints, initial_state=rnn_tuple_state, time_major=False)
        elif cell_type == 'GRU':
            with tf.name_scope('multi_GRU'):
                multicell = tf.contrib.rnn.MultiRNNCell([GRUCell(CELLSIZE) for _ in range(NLAYERS)], state_is_tuple = False)
            with tf.name_scope('RNN_output'):
                Hr, H = tf.nn.dynamic_rnn(multicell, X_subints, initial_state= Hin, time_major=False)
            init_state = Hin
        elif cell_type == 'LSTM':
            lstm_cell = rnn.BasicLSTMCell(CELLSIZE, forget_bias=1.0)
            x = tf.unstack(X_subints, CandConfig.TIME_LEN,1)
            Hr, H = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            init_state = None
        # Hr: [ BATCHSIZE, SEQLEN, CELLSIZE ]
        # H:  [ BATCHSIZE, CELLSIZE*NLAYERS ] # this is the last state in the sequence
        # Dense layer 1
        # CELLSIZE = 32

        with tf.name_scope('flatten'):
            lstm_h_pool2_flat = tf.reshape(Hr, [-1,CELLSIZE * t_len])
        # dense layer 1
        rnn_fc1_output = add_dense_layer(inputs=lstm_h_pool2_flat,n_layer='rnn_dense1',activation_func=activation_func,
                                         cnn_keep_prob=cnn_keep_prob,selu_dp_rate=selu_dp_rate,nodes_f=CELLSIZE*t_len,nodes_b=128)
        # tf.summary.histogram('rnn_fc1_output', rnn_fc1_output)
        # dense layer 2
        rnn_fc2_output = add_dense_layer(inputs=rnn_fc1_output, n_layer='rnn_dense2',
                                         activation_func=activation_func,
                                         cnn_keep_prob=cnn_keep_prob, selu_dp_rate=selu_dp_rate,
                                         nodes_f=128, nodes_b=256)
        # tf.summary.histogram('rnn_fc2_output', rnn_fc2_output)
        return rnn_fc2_output, init_state
def append_to_label_list(label_list, y):
    for on_hot_label in y:
        label_list.append(on_hot_label.tolist())
    return label_list
def evaluate(predict_lable, ground_truth_label, threshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for sample_no in range(len(predict_lable)):
        if predict_lable[sample_no][1] >= threshold and ground_truth_label[sample_no][1] == 1:
            TP += 1
        elif predict_lable[sample_no][1] >= threshold and ground_truth_label[sample_no][1] == 0:
            FP += 1
        elif predict_lable[sample_no][1] < threshold and ground_truth_label[sample_no][0] == 1:
            TN += 1
        else:
            FN += 1
    if TP+TN+FP+FN == 0:
        accuracy = 0
    else:
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP/(TP+FP)
    if TP+FN == 0:
        recall = 0
    else:
        recall = TP/(TP+FN)
    if precision+recall==0:
        F_score = 0
    else:
        F_score = (2*precision*recall)/(precision+recall)
    return [TP, TN, FP, FN], accuracy, precision, recall, F_score
def save_result_file(results):
    Desfile = '.\\results_test.csv'
    csv_writer = csv.writer(open(Desfile,'w',newline=''),dialect='excel')
    csv_writer.writerow(['lr','keep_prob','NLAYERS','epoch','threshold','TP','TN','FP','FN','accuracy','precision','recall','f_score'])
    for line in results:
        csv_writer.writerow(line)
def normalize(attr,scope=[-1,1]):
    if scope == [-1,1]:
        row, col = attr.shape
        for row_no in range(row):
            attr[row_no] = 2*(attr[row_no]-np.amin(attr))/(np.amax(attr)-np.amin(attr))-1
        return attr
    elif scope is None:
        return attr

def main():
    feature_cnn = cnn_model()
    feature_lstm,H = LSTM_model()
    # adding other attributes
    cnn_other_attr = tf.placeholder("float",shape=(None,2))
    #cnn_h_fc2 = tf.concat([cnn_h_fc1,cnn_other_attr],1)

    # final_feature = tf.concat([feature_cnn,feature_lstm,cnn_other_attr],1)
    with tf.name_scope('concat_layer'):
        final_feature = tf.concat([feature_cnn, feature_lstm], 1)
    # Dense layer
    _, cnn_output_len = feature_cnn.shape
    _, lstm_output_len = feature_lstm.shape


    results = []
    #learning_rates = [0.01]
    #keep_probs = [0.4]

    for learning_rate in learning_rates:
        for keep_prob in keep_probs:
            mixed_h_fc_output = add_dense_layer(inputs=final_feature,n_layer=3,activation_func=activation_func,
                                                cnn_keep_prob=cnn_keep_prob,selu_dp_rate=selu_dp_rate,
                                                nodes_f=int(cnn_output_len + lstm_output_len),nodes_b=256)
            # variable_summaries(mixed_h_fc_output,var_name='mixed_h_fc_output')
            # tf.summary.histogram('mixed_h_fc_output', mixed_h_fc_output)
            # output layer
            with tf.name_scope('prediction'):
                with tf.name_scope('weights'):
                    W_fc2 = weight_variable([256, 2])
                with tf.name_scope('bias'):
                    b_fc2 = bias_variable([2])
                with tf.name_scope('output'):
                    y_conv = tf.nn.softmax(tf.matmul(mixed_h_fc_output, W_fc2) + b_fc2)
#            y_conv = tf.matmul(mixed_h_fc1_drop,W_fc2)+b_fc2
#            y_conv = tf.nn.sigmoid(tf.matmul(mixed_h_fc1_drop, W_fc2) + b_fc2)

            # loss
            # y_ 为期望输出，y__conv 为实际输出
            with tf.name_scope('loss'):
                # loss_function = -tf.reduce_sum(y_ * tf.log(y_conv)+(1-y_) * tf.log(1-y_conv))
                # loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

                # avoid loss = nan
                loss_function = -tf.reduce_sum(y_*tf.log(y_conv + 1e-5))
                # loss_function = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
                # loss_function = tf.reduce_mean(tf.square(y_conv - y_))
                tf.summary.scalar('loss', loss_function)
            # var_grad = tf.gradients(loss_function, vars)
            # mean_grad = [tf.reduce_mean(tf.abs(v)) for v in var_grad]
            with tf.name_scope('train'):
                # train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
                # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
                train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss_function)
                # train_step = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss_function)
                # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_function)
            '''
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # gradients: return A list of sum(dy/dx) for each x in xs.
            tvars = tf.trainable_variables()
            grads = optimizer.gradients(loss_function,tvars)
            clipped_grads = tf.clip_by_global_norm(grads, config.max_grad_norm)
            # accept: List of (gradient, variable) pairs, so zip() is needed
            train_step = optimizer.apply_gradients(zip(grads,tvars))
            '''
            with tf.name_scope('correct_prediction'):
                correct_predictin = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
            with tf.name_scope('compute_accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_predictin,'float'))
            if CELL_TYPE == 'GRU':
                istate = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])  # initial zero input state
            elif CELL_TYPE == 'NAC':
                istate = np.zeros((NLAYERS, 2, BATCHSIZE, CELLSIZE))
            # 合并到Summary中
            merged = tf.summary.merge_all()
            # 选定可视化存储目录
            writer = tf.summary.FileWriter("..\\graph", sess.graph)
            sess.run(tf.global_variables_initializer())
            def get_fit_cnn_input(input_list):
                for sample_no in range(len(input_list)):
                    input_list[sample_no] = normalize(input_list[sample_no],scope = ATTR_NORMALIZE_SCOPE).reshape(f_len,p_len,1)
                return np.asarray(input_list)

            def turn_into_onehot(label, depth=2):
                label_onehot = []
                for Curlabel in label:
                    onehot_format = [0]*depth
                    onehot_format[Curlabel[0]] = 1
                    label_onehot.append(onehot_format)
                return np.asarray(label_onehot)
            print('Training...')
            loss_list = []
            print('n_batch:%g'%(n_batch))
            for i in range(n_batch):
                batch = train_batches.next_batch()
                cnn_input = get_fit_cnn_input(batch['attrs'][1])       # sub_bands (type = list)
                lstm_input = np.asarray(batch['attrs'][0])      # sub_ints (type = list)
                #lstm_input = lstm_input.reshape(BATCHSIZE,CandConfig.TIME_LEN,CandConfig.PHASE_LEN)
                other_attrs = []
                # profile
                profile = batch['attrs'][2]
                # bary_p1
                bary_p1 = batch['attrs'][3]
                topo_p1 = batch['attrs'][4]
                label = batch['label']
                #print(cnn_input)
                #print(lstm_input)
                label = turn_into_onehot(label,depth=CandConfig.CLASSNUM)
                if lstm_input.shape != (BATCHSIZE,CandConfig.TIME_LEN,CandConfig.PHASE_LEN):
                    lstm_input_fit = np.empty(shape=(BATCHSIZE,CandConfig.TIME_LEN,CandConfig.PHASE_LEN))
                    for i in range(BATCHSIZE):
                        Cur_value = lstm_input[i].copy()
                        Cur_value.resize(CandConfig.TIME_LEN,CandConfig.PHASE_LEN)
                        lstm_input_fit[i, :, :] = Cur_value
                    lstm_input = lstm_input_fit
                for sample_no in range(len(lstm_input)):
                    lstm_input[sample_no] = normalize(lstm_input[sample_no],scope = ATTR_NORMALIZE_SCOPE)
                if H is not None:
                    train_feed_dict = {X_subbands:cnn_input, X_subints:lstm_input,y_:label, H:istate, cnn_keep_prob:keep_prob}
                else:
                    train_feed_dict = {X_subbands:cnn_input, X_subints:lstm_input,y_:label, cnn_keep_prob:keep_prob}

                if H is not None:
                    if i%10 == 0:
                        merged_result,_, acc, y, outH, loss = sess.run([merged,train_step, accuracy, y_conv,H,loss_function ], feed_dict=train_feed_dict)
                        print("step:%d, training_accuracy: %g loss %g"%(i,acc,loss))
                        loss_list.append(loss)
                        print(y)
                        writer.add_summary(merged_result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）
                    #print(sess.run(mean_grad, feed_dict=train_feed_dict))
                    _, acc, y, outH, loss = sess.run([train_step, accuracy, y_conv, H, loss_function], feed_dict=train_feed_dict)
                    loss_list.append(loss)
                    #train_step.run(session = sess,feed_dict=feed_dict)
                    istate = outH
                else:
                    if i%10 == 0:
                        _, acc, y, loss = sess.run([train_step, accuracy, y_conv,loss_function ], feed_dict=train_feed_dict)
                        print("step:%d, training_accuracy: %g loss %g"%(i,acc,loss))
                        loss_list.append(loss)
                        print(y)
                    #print(sess.run(mean_grad, feed_dict=train_feed_dict))
                    _, acc, y,  loss = sess.run([train_step, accuracy, y_conv, loss_function], feed_dict=train_feed_dict)
                    loss_list.append(loss)
                    #train_step.run(session = sess,feed_dict=feed_dict)
                # exit()
            plt.plot(loss_list)
            plt.title('LOSS (MSE)')
            plt.xlabel('batch no')
            plt.ylabel('mse')
            #plt.show()
            #exit()
            print('Testing...')
            predict_lable = []
            ground_truth_label = []
            '''
            while True:
                test_batch = all_Candidates.test.next_batch(BATCHSIZE)
                other_attrs_test = []
                # dm
                other_attrs_test.append(test_batch[2])
                # period
                other_attrs_test.append(test_batch[3])
                other_attrs_test = np.asarray(other_attrs_test).T
                test_cnn_input = get_fit_cnn_input(test_batch[0])
                test_lstm_input = np.asarray(test_batch[1])
                test_y_ = test_batch[4]
                ground_truth_label= append_to_label_list(ground_truth_label,test_y_)
                # print(test_cnn_input.shape,np.asarray(test_lstm_input).shape,test_y_.shape)
                # test_feed_dict = {X_subbands:test_cnn_input,X_subints:test_lstm_input,Hin:istate,y_:test_y_,cnn_other_attr:other_attrs_test,cnn_keep_prob:0.5}
                test_feed_dict = {X_subbands:test_cnn_input,X_subints:test_lstm_input,Hin:istate,y_:test_y_,cnn_keep_prob:1.0}
                print('test_lstm_input:',test_lstm_input.shape)
                #acc = accuracy.eval(feed_dict = test_feed_dict)
                acc,y = sess.run([accuracy, y_conv], feed_dict = test_feed_dict)
                predict_lable = append_to_label_list(predict_lable,y)
                if all_Candidates.test.epoch_completed >=1:
                    print(acc)
                    print(len(ground_truth_label))
                    print(len(predict_lable))
                    for threshold in np.linspace(0,1,51):
                        sample_num, accuracy, precision, recall, F_score = evaluate(predict_lable, ground_truth_label, threshold)
                        print('sample_num',sample_num,'sum',sum(sample_num),'acc:',accuracy,'precision:',precision,'recall:',recall,'F_score:',F_score)
                        write_line = [learning_rate,keep_prob,NLAYERS]
                        write_line.extend([all_Candidates.test.epoch_completed,threshold,sample_num[0],sample_num[1],sample_num[2],sample_num[3],accuracy,precision,recall,F_score])
                        results.append(write_line)
                    break
            '''
    save_result_file(results)
if __name__ == '__main__':
    main()
