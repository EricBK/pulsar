import tensorflow as tf
import input_data_pulsar
import numpy as np
from tensorflow.contrib import rnn
import math
import csv
p_path = 'E:\\pg\\1-SKA\\Data\\MedlatTrainingData\\pkl_files\\pulsars_attrs_aug.pkl'
n_path = 'E:\\pg\\1-SKA\\Data\\MedlatTrainingData\\pkl_files\\RFI_attrs.pkl'
cand_file_format = 'phcx' # 'pfd'
all_Candidates = input_data_pulsar.read_data(p_path, n_path,test_part = 0.3,cand_file_format = cand_file_format)
f_len, t_len, p_len = all_Candidates.train.data_size
print('f_len:',f_len,'t_len:',t_len,'p_len:',p_len)
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.InteractiveSession(config=config)
sess = tf.InteractiveSession()

X_subbands = tf.placeholder("float",shape=(None,f_len,p_len,1))
X_subints = tf.placeholder(tf.float32, [None, t_len, p_len])  # [Batchsize,t_len, f_len]

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

cnn_keep_prob = tf.placeholder('float')

def cnn_model():
    # 第一层卷积
    kernal_width = 3
    kernal_length = 3
    cnn_W_conv1 = weight_variable([kernal_length,kernal_width,1,8])
    cnn_b_conv1 = bias_variable([8])

    cnn_h_conv1 = tf.nn.relu(conv2d(X_subbands,cnn_W_conv1) + cnn_b_conv1)
    cnn_h_pool1 = max_pool_2x2(cnn_h_conv1)

    # 第二层卷积
    cnn_W_conv2 = weight_variable([kernal_length,kernal_width,8,32])
    cnn_b_conv2 = bias_variable([32])

    cnn_h_conv2 = tf.nn.relu(conv2d(cnn_h_pool1,cnn_W_conv2) + cnn_b_conv2)
    cnn_h_pool2 = max_pool_2x2(cnn_h_conv2)

    # 第三层卷积
    cnn_W_conv3 = weight_variable([kernal_length,kernal_width,32,64])
    cnn_b_conv3 = bias_variable([64])

    cnn_h_conv3 = tf.nn.relu(conv2d(cnn_h_pool2,cnn_W_conv3) + cnn_b_conv3)
    cnn_h_pool3 = max_pool_2x2(cnn_h_conv3)

    convolution_layer = 3
    de_multi = int(math.pow(2,convolution_layer))
    # Dense layer_1
    cnn_W_fc1 = weight_variable([int(f_len/de_multi)*int(p_len/de_multi)*64,128])
    cnn_b_fc1 = bias_variable([128])

    cnn_h_pool2_flat = tf.reshape(cnn_h_pool3, [-1,int(f_len/de_multi)*int(p_len/de_multi)*64])
    cnn_h_fc1 = tf.nn.relu(tf.matmul(cnn_h_pool2_flat,cnn_W_fc1) + cnn_b_fc1)
    # dropout
    cnn_h_fc1_drop = tf.nn.dropout(cnn_h_fc1,cnn_keep_prob)

    # Dense layer_2
    cnn_W_fc2 = weight_variable([128,256])
    cnn_b_fc2 = bias_variable([256])
    cnn_h_fc2 = tf.nn.relu(tf.matmul(cnn_h_fc1_drop,cnn_W_fc2) + cnn_b_fc2)
    # dropout
    cnn_h_fc2_drop = tf.nn.dropout(cnn_h_fc2,cnn_keep_prob)

    return cnn_h_fc2_drop


# LSTM hyerparameters
ALPHASIZE = 64
CELLSIZE = p_len
NLAYERS = 3
SEQLEN = 18
BATCHSIZE = 50
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])

def LSTM_model():

    # LSTM model
    # 3层 GRU
    dropout_keep = 1.0
    lr = tf.placeholder(tf.float32, name='lr')  # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    cell = rnn.GRUCell(CELLSIZE)
    mcell = rnn.MultiRNNCell([cell]*NLAYERS, state_is_tuple = False)
    Hr, H = tf.nn.dynamic_rnn(mcell, X_subints, initial_state = Hin)
    # Hr: [ BATCHSIZE, SEQLEN, CELLSIZE ]
    # H:  [ BATCHSIZE, CELLSIZE*NLAYERS ] # this is the last state in the sequence
    print('Hr shape:',Hr.shape)
    # Dense layer 1
    lstm_W_fc1 = weight_variable([CELLSIZE*t_len,128])
    lstm_b_fc1 = bias_variable([128])
    lstm_h_pool2_flat = tf.reshape(Hr, [-1,CELLSIZE * t_len])
    lstm_h_fc1 = tf.nn.relu(tf.matmul(lstm_h_pool2_flat,lstm_W_fc1) + lstm_b_fc1)
    # dropout
    lstm_h_fc1_drop = tf.nn.dropout(lstm_h_fc1, cnn_keep_prob)
    # Dense layer 2
    lstm_W_fc2 = weight_variable([128,256])
    lstm_b_fc2 = bias_variable([256])
    lstm_h_fc2 = tf.nn.relu(tf.matmul(lstm_h_fc1_drop,lstm_W_fc2) + lstm_b_fc2)
    # dropout
    lstm_h_fc2_drop = tf.nn.dropout(lstm_h_fc2, cnn_keep_prob)
    #print('lstm_H_fc1 shape:',lstm_h_fc1.shape)
    return lstm_h_fc2_drop, H
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
    Desfile = '.\\results.csv'
    csv_writer = csv.writer(open(Desfile,'w',newline=''),dialect='excel')
    csv_writer.writerow(['lr','keep_prob','NLAYERS','epoch','threshold','TP','TN','FP','FN','accuracy','precision','recall','f_score'])
    for line in results:
        csv_writer.writerow(line)
def main():
    feature_cnn = cnn_model()
    feature_lstm,H = LSTM_model()
    # adding other attributes
    cnn_other_attr = tf.placeholder("float",shape=(None,2))
    #cnn_h_fc2 = tf.concat([cnn_h_fc1,cnn_other_attr],1)

    # final_feature = tf.concat([feature_cnn,feature_lstm,cnn_other_attr],1)
    final_feature = tf.concat([feature_cnn, feature_lstm], 1)
    # Dense layer
    _, cnn_output_len = feature_cnn.shape
    _, lstm_output_len = feature_lstm.shape


    learning_rates = [0.001, 0.0001]
    keep_probs = [0.4, 0.5, 0.6, 0.7]
    results = []
    #learning_rates = [0.01]
    #keep_probs = [0.4]
    n_batch = 2000
    for learning_rate in learning_rates:
        for keep_prob in keep_probs:
            all_Candidates.train.reset()
            all_Candidates.test.reset()
            mixed_W_fc1 = weight_variable([int(cnn_output_len + lstm_output_len), 512])
            mixed_b_fc1 = bias_variable([512])

            mixed_h_fc1 = tf.nn.relu(tf.matmul(final_feature, mixed_W_fc1) + mixed_b_fc1)
            # dropout
            mixed_h_fc1_drop = tf.nn.dropout(mixed_h_fc1, cnn_keep_prob)

            # output layer
            W_fc2 = weight_variable([512, 2])
            b_fc2 = bias_variable([2])

            y_conv = tf.nn.softmax(tf.matmul(mixed_h_fc1_drop, W_fc2) + b_fc2)

            # loss
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

            correct_predictin = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictin,'float'))
            istate = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])  # initial zero input state
            sess.run(tf.initialize_all_variables())

            def get_fit_cnn_input(input_list):
                for sample_no in range(len(input_list)):
                    input_list[sample_no] = input_list[sample_no].reshape(f_len,p_len,1)
                return np.asarray(input_list)
            print('Training...')
            for i in range(n_batch):
                batch = all_Candidates.train.next_batch(BATCHSIZE)
                cnn_input = get_fit_cnn_input(batch[0])
                lstm_input = np.asarray(batch[1])
                other_attrs = []
                # dm
                other_attrs.append(batch[2])
                # period
                other_attrs.append(batch[3])
                other_attrs = np.asarray(other_attrs).T
                # train_feed_dict = {X_subbands:cnn_input,X_subints:lstm_input,Hin:istate,y_:batch[4],cnn_other_attr:other_attrs, cnn_keep_prob:0.5}
                train_feed_dict = {X_subbands:cnn_input,X_subints:lstm_input,Hin:istate,y_:batch[4], cnn_keep_prob:keep_prob}
                '''
                moving_avg_p = 10
                Cur_acc == 0
                if i==moving_avg_p:
                    Cur_acc = get_acc(session = sess, data = all_Candidates.validation)
                '''
                if i%10 == 0:
                    _, acc, y, outH = sess.run([train_step, accuracy, y_conv, H, ], feed_dict=train_feed_dict)
                    print("step:%d, training_accuracy: %g"%(i,acc))
                    #print(outH.shape)
                _, acc, y, outH = sess.run([train_step, accuracy, y_conv, H, ], feed_dict=train_feed_dict)


                #train_step.run(session = sess,feed_dict=feed_dict)
                istate = outH
            print('Testing...')
            predict_lable = []
            ground_truth_label = []

            while True:
                test_batch = all_Candidates.test.next_batch(BATCHSIZE)
                other_attrs_test = []
                # dm
                other_attrs_test.append(test_batch[2])
                # period
                other_attrs_test.append(test_batch[3])
                other_attrs_test = np.asarray(other_attrs_test).T
                test_cnn_input = get_fit_cnn_input(test_batch[0])
                test_lstm_input = test_batch[1]
                test_y_ = test_batch[4]
                ground_truth_label= append_to_label_list(ground_truth_label,test_y_)
                # print(test_cnn_input.shape,np.asarray(test_lstm_input).shape,test_y_.shape)
                # test_feed_dict = {X_subbands:test_cnn_input,X_subints:test_lstm_input,Hin:istate,y_:test_y_,cnn_other_attr:other_attrs_test,cnn_keep_prob:0.5}
                test_feed_dict = {X_subbands:test_cnn_input,X_subints:test_lstm_input,Hin:istate,y_:test_y_,cnn_keep_prob:1.0}

                #acc = accuracy.eval(feed_dict = test_feed_dict)
                acc,y = sess.run([accuracy, y_conv], feed_dict=test_feed_dict)
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
    save_result_file(results)
if __name__ == '__main__':
    main()
