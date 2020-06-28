# 用来组织构造TFRecord文件
# from ..extend.dm3 import DM3lib as dm3
from app.extend.dm3 import DM3lib as dm3
import numpy as np
import os, random
import tensorflow as tf


class DataSet2TFRecord:
    def __init__(self, size='A'):
        self.size = size
        self.size_name_dict = {'S': ['59k', '98k'], 'M': ['200k'], 'L': ['490k']}
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.rawDir = self.rootDir + 'dataset\\tem\\raw\\local\\'
        self.desDir = self.rootDir + 'dataset\\tem\\tfrecords\\'

    def dm3_to_tfrecords(self):
        # size： S-~100k倍率，M-100k~300k倍率，L-300k~倍率，A-所有倍率
        if self.size in ['S', 'M', 'L', 'A']:
            print('开始处理数据,尺寸参数:{}'.format(self.size))
            train_tfrecords_path = self.desDir + 'train.tfrecords'
            valid_tfrecords_path = self.desDir + 'valid.tfrecords'
            test_tfrecords_path = self.desDir + 'test.tfrecords'
            file_name_list = os.listdir(self.rawDir)
            # 遍历文件夹内部所有文件名并通过随机数按照6:2:2的比例分别存入3个列表当中
            train_file_name_list = []
            valid_file_name_list = []
            test_file_name_list = []
            random.shuffle(file_name_list)
            for index in range(file_name_list.__len__()):
                std = float(index) / float(file_name_list.__len__())
                if std <= 0.6:
                    train_file_name_list.append(file_name_list[index])
                elif std <= 0.8:
                    valid_file_name_list.append(file_name_list[index])
                else:
                    test_file_name_list.append(file_name_list[index])
            self.write_tfrecords(train_file_name_list, train_tfrecords_path)
            self.write_tfrecords(valid_file_name_list, valid_tfrecords_path)
            self.write_tfrecords(test_file_name_list, test_tfrecords_path)

        else:
            raise ValueError('别瞎鸡儿输入尺寸参数:{}'.format(self.size))

    def write_tfrecords(self, file_name_list, tfrecords_path):
        with tf.io.TFRecordWriter(tfrecords_path, options=None) as writer:
            print('开始写入tfrecords')
            for file_name in file_name_list:
                if self.size == 'A' or file_name.split('-')[1].replace('BF', '') in self.size_name_dict[self.size]:
                    print(file_name)
                    dm3f = dm3.DM3(self.rawDir + file_name)
                    imgd = dm3f.imagedata  # (2048, 2048) float32的2维
                    # print(imgd.shape, imgd.dtype)
                    # print(imgd.numpy())
                    # 开始写入tfrecords文件
                    exam = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'name': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[file_name.encode('utf-8')])),
                                'label': tf.train.Feature(int64_list=tf.train.Int64List(
                                    value=[int(file_name.split('-')[2].replace('.dm3', ''))])),
                                'shape': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[imgd.shape[0], imgd.shape[1]])),
                                'data': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=imgd.reshape(-1)))
                            }
                        )
                    )
                    writer.write(exam.SerializeToString())


if __name__ == '__main__':
    DataSet2TFRecord(size='M').dm3_to_tfrecords()
