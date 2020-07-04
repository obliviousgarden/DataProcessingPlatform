import tensorflow as tf
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.core import serializers
import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


@require_http_methods(["GET"])
def tf_tem(request):
    response = {}
    print(request.GET)
    # TODO：调取数据库信息对图片进行迁移分析
    response['code'] = 738
    response['data'] = {'data': ''}
    response['msg'] = '分析成功'
    return JsonResponse(response)


# 下面是神经网络相关代码
# 1 数据
def parse_example(example):
    features = tf.io.parse_single_example(example,
                                          features={'name': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64),
                                                    'shape': tf.io.FixedLenFeature([2], tf.int64),
                                                    'data': tf.io.FixedLenFeature([2048, 2048], tf.float32)})
    name = features['name']
    label = features['label']
    shape = features['shape']
    data = features['data']
    # data_raw = np.fromstring(features['data'], dtype=np.float32)
    # reshaped_data = np.reshape(data_raw, features['shape'])
    # data = (reshaped_data - np.mean(reshaped_data, axis=0)) / np.std(reshaped_data, axis=0)
    return name, label, shape, data


class DataLoader():
    def __init__(self):
        print('初始化DataLoader，载入数据中...')
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.datasetDir = self.rootDir + 'dataset\\tem\\tfrecords\\'
        self.train_dataset = tf.data.TFRecordDataset(self.datasetDir + 'train.tfrecords')
        self.valid_dataset = tf.data.TFRecordDataset(self.datasetDir + 'valid.tfrecords')
        self.test_dataset = tf.data.TFRecordDataset(self.datasetDir + 'test.tfrecords')

    def get_dataset(self, filename):
        if filename is 'train':
            return self.train_dataset.map(parse_example)
        elif filename is 'valid':
            return self.valid_dataset.map(parse_example)
        else:
            return self.test_dataset.map(parse_example)

    def get_batch(self, filename, epochs, shuffle, batch_size):
        return self.get_dataset(filename).repeat(epochs).shuffle(shuffle).batch(batch_size)


# 2 模型
class UNet(tf.keras.Model):
    def __init__(self, input_size=(2048, 2048, 1), pretrained_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(pretrained_weights, input_size)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            name='encoder_conv1_1'
        )

    def call(self, inputs):
        print(inputs)
        output = {}
        return output


# 3 训练
num_epochs = 10  # 1个epoch表示过了 1遍训练集中的所有样本
batch_size = 20  # 1次迭代中的样本数量
learning_rate = 0.001  # 学习速率

# 4 评价

if __name__ == '__main__':
    data_loader = DataLoader()
    batch = data_loader.get_batch('train', num_epochs, 1000, batch_size)
    # TODO：设法取出迭代器开始迭代，不过需要提前把MODEL设计好
    print(batch)
    # <BatchDataset shapes: ((None,), (None,), (None, 2), (None, 2048, 2048)), types: (tf.string, tf.int64, tf.int64, tf.float32)>
