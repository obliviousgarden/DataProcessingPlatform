import os, random
import tensorflow as tf


class DataSet2TFRecord:
    def __init__(self):
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.rawDir = self.rootDir + 'dataset\\biasdrivendielectric\\raw\\'
        self.desPath = self.rootDir + 'dataset\\biasdrivendielectric\\tfrecords\\train.tfrecords'

    def raw_to_tfrecords(self):
        file_name_list = os.listdir(self.rawDir)
        with tf.io.TFRecordWriter(self.desPath, options=None) as writer:
            print('开始写入TFRecords')
            for file_name in file_name_list:
                sample_serial, dc_bias = file_name.split('-DCB')
                print(sample_serial, dc_bias)
                with open(self.rawDir + file_name, 'r') as file:
                    lines = file.readlines()
                    thickness = float(lines[4])
                    electrode_area = float(lines[5])
                    freq_list = []
                    epsilon_list = []
                    for line_index in range(19, lines.__len__()):
                        freq, loss, c = lines[line_index].replace('\n', '').split('\t')
                        freq_list.append(float(freq))
                        epsilon_list.append(float(c) * thickness * 10000000000 / (8.854 * electrode_area))
                    print(thickness)
                    print(electrode_area)
                    print(freq_list)
                    print(epsilon_list)
                    exam = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'label': tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[sample_serial.encode('utf-8')])
                                ),
                                'dc_bias': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[float(dc_bias)])
                                ),
                                'thickness': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[thickness])
                                ),
                                'electrode_area': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=[electrode_area])
                                ),
                                'freq_list': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=freq_list)
                                ),
                                'epsilon_list': tf.train.Feature(
                                    float_list=tf.train.FloatList(value=epsilon_list)
                                ),
                            }
                        )
                    )
                    writer.write(exam.SerializeToString())


if __name__ == '__main__':
    DataSet2TFRecord().raw_to_tfrecords()
