from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.core import serializers
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from sklearn import linear_model


class DataLoader():
    def __init__(self):
        print('数据初始化中...')
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.rawDir = self.rootDir + 'dataset\\biasdrivendielectric\\raw\\'

    def get_data(self, dc_bias):
        file_name_list = os.listdir(self.rawDir)
        freq_list = []
        epsilon_list = []
        for file_name in file_name_list:
            if dc_bias == float(file_name.split('-DCB')[1]):
                with open(self.rawDir + file_name, 'r') as file:
                    lines = file.readlines()
                    thickness = float(lines[4])
                    electrode_area = float(lines[5])
                    for line_index in range(19, lines.__len__()):
                        freq, loss, c = map(float, lines[line_index].replace('\n', '').split('\t'))
                        # FIXME:暂时把第一个介电弛豫阶段排除
                        if freq >= 1000.0:
                            freq_list.append(freq)
                            epsilon_list.append(c * thickness * 10000000000 / (8.854 * electrode_area))
                break
        return freq_list, epsilon_list


class BiasDrivenDielectricModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
        self.dense_2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        inputs = tf.keras.layers.Input(inputs)
        output = self.dense(inputs)
        return output


def func(omega, epsilon_inf, delta_epsilon, tau):
    # return epsilon_inf + delta_epsilon / (1 + tf.square(tf.multiply(omega, tau)))
    return tf.square(tf.multiply(omega, tau))


def main(dc_bias):
    # START:利用现行回归对colecole拟合成功，暂时封上
    # data_loader = DataLoader()
    # freq_test, epsilon_test = np.array(data_loader.get_data(dc_bias), dtype='float64')
    # epsilon_0 = 530.0
    # epsilon_inf = 50.0
    # # epsilon_0 = np.max(epsilon_test)
    # # epsilon_inf = np.min(epsilon_test)
    # print((epsilon_0 - epsilon_inf) / (epsilon_test - epsilon_inf) - 1.0)
    # y = []
    # x = []
    # for i in range(freq_test.size):
    #     y_value = (epsilon_0 - epsilon_inf) / (epsilon_test[i] - epsilon_inf) - 1.0
    #     if y_value > 0.0:
    #         x.append(2.0 * np.log10(freq_test[i]))
    #         y.append(np.log10(y_value))
    # model = linear_model.LinearRegression()
    # model.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
    #
    # print('线性回归：',model.intercept_[0],model.coef_[0][0])
    # intercept =model.intercept_[0]
    # coef = model.coef_[0][0]
    # y_fit = intercept+np.multiply(x,coef)
    # #intercept = 2log(c),c=2*pi*tm
    # epsilon_fit = (epsilon_0-epsilon_inf)/(np.power(10,intercept+np.multiply(2*np.log10(freq_test),coef))+1.0)+epsilon_inf
    # c = np.power(10,intercept/2)
    # tau = np.power(10,intercept/2)/(2*np.pi)
    # # epsilon_fit = epsilon_inf+(epsilon_0-epsilon_inf)/(1+np.square(np.multiply(c,freq_test)))
    # print('tau',tau)
    # fig = plt.figure()
    # ax_211 = fig.add_subplot(2, 1, 1)
    # ax_211.scatter(x, y)
    # ax_211.plot(x,y_fit,'r-')
    # plt.ion()
    # plt.show()
    # ax_212 = fig.add_subplot(2,1,2)
    #
    # ax_212.scatter(freq_test, epsilon_test)
    # ax_212.set_xscale('log')
    # ax_212.plot(freq_test,epsilon_fit,'r-')
    # # END:利用现行回归对colecole拟合成功，暂时封上
    # START：使用BP神经网络对Havriliak–Negami Model

    data_loader = DataLoader()
    freq_test, epsilon_test = np.array(data_loader.get_data(dc_bias), dtype='float32')
    print('读取数据成功,频率数据集:{},介电数据集:{}'.format(freq_test.shape, epsilon_test.shape))
    freq_test = freq_test[:, np.newaxis]
    omega_test = np.multiply(freq_test, 2.0 * np.pi)
    epsilon_test = epsilon_test[:, np.newaxis]
    print('reshape维数到[None,1],频率数据集:{},介电数据集:{}'.format(freq_test.shape, epsilon_test.shape))
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax_331 = fig.add_subplot(3, 3, 1)
    ax_331.set_xscale('log')
    ax_331.scatter(freq_test, epsilon_test, label='epsilon(freq)')
    ax_331.legend()

    x_target = np.log(omega_test + np.random.normal(0.0, 10.0, omega_test.shape))
    W1_target = 0.9
    b1_target = 0.9 * np.log(0.000001)
    y_apos_apos_target = np.multiply(x_target, W1_target) + b1_target + + np.random.normal(0.0, 0.1, x_target.shape)
    ax_332 = fig.add_subplot(3, 3, 2)
    ax_332.set_xscale('log')
    ax_332.scatter(omega_test, y_apos_apos_target, label='y\'\'(omega)_T')
    ax_332.legend()
    # FIXME:需要记录一下极值帮助f_apos在输入时的归一化
    f_apos_target = np.exp(y_apos_apos_target) + np.random.normal(0.0, 0.1, y_apos_apos_target.shape)
    f_apos_target_max = np.max(f_apos_target)
    f_apos_target_min = np.min(f_apos_target)
    f_apos_target_mean= np.mean(f_apos_target)
    f_apos_target_std = np.std(f_apos_target)
    y_apos_target = 1.0 + np.square(f_apos_target) - 2.0 * f_apos_target * np.cos(0.9 * np.pi / 2.0) + np.random.normal(
        0.0, 0.1, f_apos_target.shape)
    ax_333 = fig.add_subplot(3, 3, 3)
    ax_333.scatter(omega_test, y_apos_target, label='y\'(omega)_T')
    ax_333.legend()

    # print(np.any(np.isnan(y_apos_target)))

    tf.compat.v1.disable_eager_execution()
    omega = tf.compat.v1.placeholder(tf.float32, [None, 1], name='INPUT_omega')
    epsilon = tf.compat.v1.placeholder(tf.float32, [None, 1], name='OUTPUT_epsilon')  # 计算途中它会代表各种各样的量

    # 层1 omega(150,1)->x(150,1)
    x = tf.math.log(omega)
    # 层2 x(150,1)->y''(150,16)
    W1 = tf.Variable(tf.compat.v1.truncated_normal([1, 16], mean=0.9, stddev=0.06, dtype='float32', name='W1_alpha'))
    b1 = tf.Variable(
        tf.compat.v1.truncated_normal([1, 16], mean=-15.0, stddev=10.0, dtype='float32', name='b1_alpha_LN_tau'))
    Wx_plus_b1 = tf.matmul(x, W1) + b1  # (150,16)
    y_apos_apos = Wx_plus_b1  # 线性激活函数

    # START:输出层(第一次线性规划使用)
    W1_out = tf.Variable(tf.divide(tf.ones([16, 1]), 16))
    b1_out = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_b1_out = tf.matmul(y_apos_apos, W1_out) + b1_out
    output_1 = Wx_plus_b1_out
    # END

    # 层3 y''(150,16)->f'(150,16)
    # FIXME:这里可能需要归一化，防止loss下降太慢
    # 普通归一化或者0均值归一化
    f_apos = tf.math.exp(y_apos_apos)
    # TODO：0均值归一化

    # 层4 f'(150,16)->y'(150,64)
    W21 = tf.Variable(
        tf.divide(tf.ones([16, 16]), 1), name='W21_1')
    W22 = tf.Variable(
        tf.compat.v1.truncated_normal([16, 16], mean=-1.8, stddev=0.11, dtype='float32', name='W22_-2_COS_alpha_PI_/2'))
    b2 = tf.Variable(tf.ones([1, 16], name='b2_1'))
    # Wx_plus_b2 = tf.matmul(tf.square(f_apos), W21) + tf.matmul(f_apos, W22) + b2
    # 归一化会导致这里的平方值出现不预期的误差
    Wx_plus_b2 = tf.matmul(tf.square(f_apos), W21) + tf.matmul(f_apos, W22) + b2
    y_apos = tf.tanh(Wx_plus_b2)
    # y_apos = tf.nn.sigmoid(Wx_plus_b2)

    # START:输出层(第二次二次规划规划使用)
    W2_out = tf.Variable(tf.divide(tf.ones([16, 1]), 16))
    b2_out = tf.Variable(tf.ones([1, 1]))
    Wx_plus_b2_out = tf.matmul(y_apos, W2_out) + b2_out
    output_2 = Wx_plus_b2_out
    # END

    # loss = tf.reduce_mean(tf.square(y_apos_apos_target - output_1))
    loss = tf.reduce_mean(tf.square(y_apos_target - output_2))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(loss)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)
        for _ in range(10000):
            # ！！！注意此处是epsilon不是output
            # sess.run(optimizer, feed_dict={omega: omega_test, epsilon: y_apos_apos_target})
            sess.run(optimizer, feed_dict={omega: omega_test, epsilon: y_apos_target})
            if _ % 50 == 0:
                # print(sess.run(loss, feed_dict={omega: omega_test, epsilon: y_apos_apos_target}))
                print(sess.run(loss, feed_dict={omega: omega_test, epsilon: y_apos_target}))
        # 获得预测值
        # y_apos_apos_pred = sess.run(output_1, feed_dict={omega: omega_test})
        y_apos_pred = sess.run(output_2, feed_dict={omega: omega_test})
    # ax_332.plot(omega_test, y_apos_apos_pred, 'r-', label='y\'\'(omega)_P')
    # ax_332.legend()
    ax_333.plot(omega_test, y_apos_pred, 'r-', label='y\'(omega)_P')
    ax_333.legend()

    # # 中间层
    # W1 = tf.Variable(tf.compat.v1.random_normal([1, 10]))
    # b1 = tf.Variable(tf.compat.v1.zeros([1, 10]))
    # Wx_plus_b1 = tf.matmul(x, W1) + b1
    # l1 = tf.nn.tanh(Wx_plus_b1)
    # # 输出层
    # W2 = tf.Variable(tf.compat.v1.random_normal([10, 1]))
    # b2 = tf.Variable(tf.compat.v1.zeros([1, 1]))
    # Wx_plus_b2 = tf.matmul(l1, W2) + b2
    # pred = tf.nn.tanh(Wx_plus_b2)
    # # 定义loss函数
    # loss = tf.reduce_mean(tf.square(y - pred))
    # cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.000000001).minimize(loss)
    # # 变量初始化
    # init = tf.compat.v1.global_variables_initializer()
    #
    # with tf.compat.v1.Session() as sess:
    #     sess.run(init)
    #     for _ in range(3000):
    #         sess.run(optimizer, feed_dict={x: freq_test, y: y_test})
    #     # 获得预测值
    #     pred_value = sess.run(pred, feed_dict={x: freq_test})
    #
    # plt.figure()
    # plt.scatter(freq_test, y_test)
    # plt.plot(freq_test, pred_value, 'r-', lw=3)
    # plt.show()
    # END

    # Keras 不是很好用 结果都是常数
    # data_loader = DataLoader()
    # freq_test, epsilon_test = tf.constant(
    #     np.array(
    #         data_loader.get_data(dc_bias), dtype='float32'))
    # print(freq_test, epsilon_test)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.scatter(freq_test, epsilon_test)
    # ax.set_xscale('log')
    # plt.ion()
    # plt.show()
    #
    # num_epoch = 1000
    #
    # model = Sequential()
    # model = Sequential()
    # model.add(Dense(units=1, activation='linear', input_shape=[1]))
    # model.add(Dense(units=64, activation='tanh'))
    # model.add(Dense(units=64, activation='tanh'))
    # model.add(Dense(units=1, activation='linear'))
    # model.compile(loss='mse', optimizer="adam")
    #
    # adam = tf.keras.optimizers.Adam(learning_rate=0.00000001)
    # model.compile(optimizer=adam, loss='mse')
    #
    # model.summary()
    #
    # epsilon_pred = func(freq_test, 20.0, 500.0, 0.000001)
    #
    # ax.plot(freq_test, epsilon_pred, 'r-', lw=5)
    # plt.pause(0.1)
    # model.fit(freq_test,
    #           epsilon_pred,
    #           epochs=num_epoch)
    # y_pred = model.predict(freq_test)
    # print(y_pred)
    # ax.plot(freq_test, y_pred, 'g-', lw=5)
    # plt.pause(0.1)


if __name__ == '__main__':
    main(0.0)
