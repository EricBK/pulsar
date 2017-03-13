from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Merge
from phcx import *
import numpy as np
import os
from keras.utils import np_utils, generic_utils

def lenet5(CNN_INPUT_length, CNN_INPUT_width):
    #首先训练一个lenet5模型，训练数据集使用subbands,并将参数保存下来
    model = Sequential()

    model.add(Convolution2D(4, 5, 5, border_mode='valid',
                            input_shape=(1, CNN_INPUT_length, CNN_INPUT_width)))
    # 第一个卷积层，4个卷积核，每个卷积核5*5,卷积后24*24，第一个卷积核要申明input_shape(通道，大小)
    model.add(Activation('tanh'))  # 激活函数采用“tanh”

    model.add(Convolution2D(8, 3, 3, subsample=(2, 2),
                            border_mode='valid'))
    # 第二个卷积层，8个卷积核，不需要申明上一个卷积留下来的特征map，会自动识别，下采样层为2*2,卷完且采样后是11*11
    model.add(Activation('tanh'))

    model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='valid'))
    # 第三个卷积层，16个卷积核，下采样层为2*2,卷完采样后是4*4
    model.add(Activation('tanh'))

    model.add(Flatten())  # 把多维的模型压平为一维的，用在卷积层到全连接层的过度
    model.add(Dense(128, input_dim=(16 * 4 * 4), init='normal'))
    # 全连接层，首层的需要指定输入维度16*4*4,128是输出维度，默认放第一位
    model.add(Activation('tanh'))

    model.add(Dense(64, input_dim=128, init='normal'))
    # 第二层全连接层，其实不需要指定输入维度，输出为10维，因为是10类
    # model.add(Activation('softmax'))  # 激活函数“softmax”，用于分类
    return model
def get_data(filePath,mode):
    if mode == 'train':
        pulsar_file_base = filePath + 'pulsars_train\\'
        rfi_file_base = filePath + 'RFI_train\\'
    else:
        pulsar_file_base = filePath + 'pulsars_test\\'
        rfi_file_base = filePath + 'RFI_test\\'
    pulsar_files = os.listdir(pulsar_file_base)
    rfi_files = os.listdir(rfi_file_base)
    cnn_input = np.empty((len(pulsar_files)+len(rfi_files), 1, 16, 64), dtype='float32')
    lstm_input = np.empty((len(pulsar_files)+len(rfi_files), 18, 64), dtype='float32')
    train_label = [1]*len(pulsar_files)
    train_label.extend([0]*len(rfi_files))
    trainlabel = np_utils.to_categorical(train_label, 2)
    train_num = 0
    for filename in pulsar_files:
        cand = Candidate(pulsar_file_base + filename)
        cnn_input[train_num,:,:,:] = np.resize(cand.subbands,(16,64))
        lstm_input[train_num,:,:] = np.resize(cand.subints,(18,64))
        train_num +=1
    for filename in rfi_files:
        cand = Candidate(rfi_file_base + filename)
        cnn_input[train_num,:,:,:] = np.resize(cand.subbands,(16,64))
        lstm_input[train_num,:,:] = np.resize(cand.subints,(18,64))
        train_num +=1
    return cnn_input,lstm_input,trainlabel
def main():
    model_cnn = lenet5(16,64)
    model_lstm = Sequential()
    model_lstm.add(LSTM(128, input_dim=64, input_length=18, dropout_W=0.2, dropout_U=0.2, consume_less="mem"))  # try using a GRU instead, for fun
    model_lstm.add(Dense(64))
    merged = Merge([model_cnn,model_lstm],mode='concat')
    model = Sequential()
    model.add(merged)
    model.add(Dense(2,activation='softmax'))

    # try using different optimizers and different optimizer configs
    # optimizer_ins = keras.optimizers.Adam(lr=0.01)
    optimizer_ins = SGD(lr=0.01, momentum=0.9, decay=0.00, nesterov=False)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer_ins,
              metrics=['accuracy'])
    model_lstm.summary()
    print('LSTM training...')

    filePath = 'H:\\研究生\\1-SKA\\Data\\MedlatTrainingData\\'
    train_data_cnn,train_data_lstm,train_label = get_data(filePath,'train')
    model.fit([train_data_cnn,train_data_lstm], train_label, batch_size=32, nb_epoch=1)
    test_data_cnn,test_data_lstm,test_label = get_data(filePath,'test')
    model.evaluate([test_data_cnn,test_data_lstm],test_label,batch_size = 16)
if __name__ == '__main__':
    main()
