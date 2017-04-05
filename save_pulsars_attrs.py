# coding = utf-8
# 需要用 python2
import h5py
import numpy as np
import os
from PFDFile import PFD
from phcx import Candidate
import matplotlib.pyplot as plt
import pickle
import random
HORIZONTAL_AUG_NUM = 16
VERTICAL_AUG_NUM = 64
def add_noise(data):
    for row_no in range(len(data)):
        for pixcel_no in range(len(data[row_no])):
            if data[row_no][pixcel_no] < 1:
                data[row_no][pixcel_no] = random.uniform(0,0.1)
            elif data[row_no][pixcel_no] == 255:
                data[row_no][pixcel_no] -= random.uniform(0,0.1)
    return data
def change(data,length,width,is_add_noise = True):
    new_data = []
    data_length,data_width = data.shape
    if data_width!=width:
        interval_width = int(width/data_width)
        for row_no in range(len(data)):
            new_row = []
            for width_no in range(data_width):
                new_row.extend([data[row_no][width_no]]*interval_width)
            if len(new_row)!=width:
                new_row.extend([new_row[-1]]*(width-len(new_row)))
            new_data.append(new_row)
        if is_add_noise:
            new_data = add_noise(new_data)
        return np.asarray(new_data)
    else:
        if is_add_noise:
            new_data = add_noise(data)
            return new_data
        else:
            return data
def horizontal_aug(data):
    extend_dt = []
    e_num = 0
    row_num, col_num = data.shape
    data = data.tolist()
    for i in range(HORIZONTAL_AUG_NUM):
        temp = []
        last_row = data.pop()
        data.insert(0,last_row)
        for ele in data:
            temp.append(ele)
        extend_dt.append(np.asarray(temp))
        e_num += 1
    assert row_num>=HORIZONTAL_AUG_NUM
    return extend_dt, e_num
def vertical_aug(data):
    extend_dt = []
    e_num = 0
    data = data.T
    row_num, col_num = data.shape
    data = data.tolist()
    for i in range(VERTICAL_AUG_NUM):
        temp = []
        last_row = data.pop()
        data.insert(0,last_row)
        for ele in data:
            temp.append(ele)
        extend_dt.append(np.asarray(temp).T)
        e_num += 1
    assert row_num>=VERTICAL_AUG_NUM
    return extend_dt, e_num
def augmentate(data, horizontal=True, vertical=True):
    extend_data = []
    extend_num = 0
    if horizontal:
        horizontal_aug_data,horizontal_aug_num  = horizontal_aug(data)
        extend_data.extend(horizontal_aug_data)
        extend_num += horizontal_aug_num
    if vertical:
        vertical_aug_data, vertical_aug_num = vertical_aug(data)
        extend_data.extend(vertical_aug_data)
        extend_num += vertical_aug_num
    return extend_data, extend_num
def save_pulsar_attrs(cand_file_path, file_fotmat,data_augmentation = False):
    if  '.pfd' in file_format:
        child_dirs = ['p309p_pfd', 'p309n_pfd']
    elif file_format == 'phcx':
        child_dirs = ['pulsars_','RFI']
        f_len = 16
        t_len = 18
        p_len = 64
    for child_dir in child_dirs:
        if not os.path.exists(cand_file_path+child_dir+'_attrs.pkl'):
            with open(cand_file_path+child_dir+'_attrs_aug.pkl','wb') as file:
                sub_bands = []
                sub_ints = []
                bestdm = []
                cand_snr = []
                cand_tp = []
                cand_num = 0
                for cand_name in os.listdir(cand_file_path+child_dir):

                    if file_format in cand_name[-4:] :
                        cand_num += 1
                        print (child_dir,cand_num)
                        if 'pfd' in file_format:
                            cand = PFD(cand_file_path+child_dir+'\\'+cand_name)
                            sub_bands.append(change(cand.get_subbands(is_scaled=True),f_len,p_len,is_add_noise=True))
                       #     print(change(cand.get_subbands(is_scaled=True),f_len,p_len).shape)
                            sub_ints.append(change(cand.get_subints(is_scaled=True),t_len,p_len,is_add_noise=True))
                       #     print(change(cand.get_subints(is_scaled=True),t_len,p_len).shape)
                            bestdm.append(cand.bestdm)
                            #cand_snr.append(cand.)
                            cand_tp.append(cand.topo_p1)
                        elif 'phcx' in file_format:
                            cand = Candidate(cand_file_path+child_dir+'\\'+cand_name)
                            sub_bands.append(cand.subbands)
                            if data_augmentation and 'pulsar' in child_dir:
                                extend_data, extend_num = augmentate(cand.subbands,horizontal = True, vertical=True)
                                sub_bands.extend(extend_data)

                            Cur_subints = cand.subints
                            Cur_subints_row, Cur_subints_col = Cur_subints.shape
                            Cur_subints = Cur_subints[:t_len,:p_len]
                            sub_ints.append(Cur_subints)

                            if data_augmentation and 'pulsar' in child_dir:
                                extend_data, extend_num = augmentate(Cur_subints, horizontal=True, vertical=True)
                                sub_ints.extend(extend_data)

                            bestdm.append(cand.dm)
                            if data_augmentation and 'pulsar' in child_dir:
                                bestdm.extend([cand.dm]*extend_num)

                            cand_tp.append(cand.topo_period)
                            if data_augmentation and 'pulsar' in child_dir:
                                cand_tp.extend([cand.topo_period]*extend_num)

                            cand_snr.append(cand.snr)
                            if data_augmentation and 'pulsar' in child_dir:
                                cand_snr.extend([cand.snr]*extend_num)
                print(len(sub_bands),len(sub_ints),len(bestdm),len(cand_tp),len(cand_snr))
                for ele in sub_bands:
                    ele_row, ele_col = ele.shape
                    if ele_row!=f_len or ele_col!= p_len:
                        print('row:',ele_row,'col:',ele_col)
                for ele in sub_ints:
                    ele_row, ele_col = ele.shape
                    if ele_row != t_len or ele_col != p_len:
                        print('row:', ele_row, 'col:', ele_col)
                pickle.dump(sub_bands,file)
                pickle.dump(sub_ints,file)
                pickle.dump(bestdm,file)
                if 'pfd' in file_format:
                    pickle.dump(cand_tp,file,-1)
                elif 'phcx' in file_format:
                    pickle.dump(cand_tp,file)
                    pickle.dump(cand_snr,file,-1)
            file.close()
if __name__ == '__main__':
    cand_file_path = 'E:\\pg\\1-SKA\\Data\\MedlatTrainingData\\'
    #cand_file_path = unicode(cand_file_path, 'utf8')

    p_len = 64
    f_len = 32
    t_len = 64
    file_format = 'phcx'
    save_pulsar_attrs(cand_file_path,file_format,data_augmentation = True)
