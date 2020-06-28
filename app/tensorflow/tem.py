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
class DataLoader():
    def __init__(self):
        print('初始化DataLoader，载入数据中...')
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.datasetDir = self.rootDir + 'dataset\\tem\\tfrecords\\'
        self.train_dataset = tf.data.TFRecordDataset(self.datasetDir+'train.tfrecords')
        self.valid_dataset = tf.data.TFRecordDataset(self.datasetDir+'valid.tfrecords')
        self.test_dataset = tf.data.TFRecordDataset(self.datasetDir + 'test.tfrecords')

        for element in self.test_dataset.as_numpy_iterator():
            features = tf.io.parse_single_example(element,
                                                  features={'name': tf.io.FixedLenFeature([], tf.string),
                                                            'label': tf.io.FixedLenFeature([], tf.int64),
                                                            'shape': tf.io.FixedLenFeature([2], tf.int64),
                                                            'data': tf.io.FixedLenFeature([2048,2048], tf.float32)})
            print(features['name'], features['label'], features['shape'], features['data'])
            img_data = np.fromstring(features['data'], dtype=np.float32)
            image_data = np.reshape(img_data, features['shape'])
            image_std_data = (image_data-np.mean(image_data,axis=0))/np.std(image_data,axis=0)
            plt.figure()
            img = plt.imshow(image_std_data,vmin=-1.0,vmax=1.0, interpolation='nearest')
            img.set_cmap('gray')
            plt.axis('off')
            plt.margins(0, 0)
            plt.show()
            print('a')


    def get_batch(self, batch_size):
        return batch_size


# 2 模型
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init()

    def call(self, inputs):
        print(inputs)
        output = {}
        return output


# 3 训练

# 4 评价

if __name__ == '__main__':
    data_loader = DataLoader()
