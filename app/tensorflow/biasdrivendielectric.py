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
from scipy.optimize import curve_fit


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


def main(dc_bias):
    # START：使用BP神经网络对Havriliak–Negami Model

    data_loader = DataLoader()
    freq_raw, epsilon_raw = np.array(data_loader.get_data(dc_bias), dtype='float32')
    print('读取数据成功,频率数据集:{},介电数据集:{}'.format(freq_raw.shape, epsilon_raw.shape))
    freq_test = freq_raw[:, np.newaxis]
    omega_test = np.multiply(freq_test, 2.0 * np.pi)
    epsilon_test = epsilon_raw[:, np.newaxis]
    alpha_target = 0.90
    d_epsilon_raw = np.gradient(epsilon_raw)
    d_epsilon_raw_max = np.max(d_epsilon_raw)
    index = 0.0
    for i in range(d_epsilon_raw.size):
        if d_epsilon_raw[i] == d_epsilon_raw_max:
            index = i
            break
    # TODO:这里的弛豫时间非常难确定 需要确定一个策略！！！！！
    tau_target = 1.0 / (np.power(2 * np.pi, 2) * omega_test[index])
    print('弛豫时间目标:{}'.format(tau_target))
    beta_target = 0.80
    epsilon_inf_target = np.min(epsilon_test)
    delta_epsilon_target = np.max(epsilon_test) - epsilon_inf_target

    print('reshape维数到[None,1],频率数据集:{},介电数据集:{}'.format(freq_test.shape, epsilon_test.shape))
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax_531 = fig.add_subplot(5, 3, 1)
    ax_531.set_xscale('log')
    ax_531.scatter(freq_test, epsilon_test, label='epsilon(freq)_T')
    ax_531.legend()

    x_target = np.log(omega_test)
    W1_target = alpha_target
    b1_target = alpha_target * np.log(tau_target)
    y_apos_apos_apos_target = np.multiply(x_target, W1_target) + b1_target
    ax_534 = fig.add_subplot(5, 3, 4)
    ax_534.set_xscale('log')
    ax_534.scatter(omega_test, y_apos_apos_apos_target, label='y\'\'\'(omega)_T')
    ax_534.legend()

    # 下面是BP神经网络
    tf.compat.v1.disable_eager_execution()

    # 第一个线性变换的BP网络
    omega_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    y_apos_apos_apos_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    # omega(150,1)->x(150,1)
    x = tf.math.log(omega_train)
    # x(150,1)->y'''(150,64)
    W1 = tf.Variable(tf.compat.v1.truncated_normal([1, 64], mean=alpha_target, stddev=1.0, dtype='float32'))
    b1 = tf.Variable(
        tf.compat.v1.truncated_normal([1, 64], mean=tf.math.log(tau_target), stddev=5.0, dtype='float32'))
    Wx_plus_b1 = tf.matmul(x, W1) + b1  # (150,16)
    y_apos_apos_apos = Wx_plus_b1  # 线性激活函数
    # y'''(150,64)->output1(150,1)
    W1_out = tf.Variable(tf.divide(tf.ones([64, 1]), 64))
    b1_out = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_b1_out = tf.matmul(y_apos_apos_apos, W1_out) + b1_out
    output_1 = Wx_plus_b1_out
    loss = tf.reduce_mean(tf.square(y_apos_apos_apos_target - output_1))
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=output_1, logits=y_apos_apos_apos_target)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.0000005).minimize(loss)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess,ui_type="readline")
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)
        for _ in range(10000):
            sess.run(optimizer, feed_dict={omega_train: omega_test, y_apos_apos_apos_train: y_apos_apos_apos_target})
            if _ % 1000 == 0:
                print('loss1', sess.run(loss, feed_dict={omega_train: omega_test,
                                                         y_apos_apos_apos_train: y_apos_apos_apos_target}))
        y_apos_apos_apos_pred = sess.run(output_1, feed_dict={omega_train: omega_test})
        W1_pred = sess.run(W1, feed_dict={omega_train: omega_test})
        b1_pred = sess.run(b1, feed_dict={omega_train: omega_test})
        W1_pred_mean = tf.reduce_mean(W1_pred).eval()
        b1_pred_mean = tf.reduce_mean(b1_pred).eval()
        alpha_pred = W1_pred_mean
        tau_pred = np.exp(b1_pred_mean / alpha_pred)
    ax_534.plot(omega_test, y_apos_apos_apos_pred, 'r-', label='y\'\'\'(omega)')
    ax_535 = fig.add_subplot(5, 3, 5)
    ax_536 = fig.add_subplot(5, 3, 6)
    ax_535.hist(W1_pred[0], rwidth=0.9, label='W1_pred_mean=' + str(W1_pred_mean))
    ax_536.hist(b1_pred[0], rwidth=0.9, label='b1_pred_mean=' + str(b1_pred_mean))
    ax_534.legend()
    ax_535.legend()
    ax_536.legend()
    print('第一组神经网络拟合完成，权值W1：{}，偏置b1：{}，换算得到系数alpha：{}，弛豫时间tau：{}'
          .format(W1_pred_mean, b1_pred_mean, alpha_pred, tau_pred))
    print('这里的弛豫时间非常不可信，需要对弛豫时间单独进行拟合计算')

    # 第二个BP的目标
    f_apos_apos_target = np.exp(y_apos_apos_apos_pred)
    y_apos_apos_target = f_apos_apos_target - np.cos(alpha_pred * np.pi / 2.0)
    f_apos_target = np.square(y_apos_apos_target)
    y_apos_target = f_apos_target + 1 - np.square(np.cos(alpha_pred * np.pi / 2))
    f_target = 0.5 * np.log(y_apos_target)
    y_target = f_target * beta_target

    ax_537 = fig.add_subplot(5, 3, 7)
    ax_537.scatter(f_target, y_target, label='y(f)_T')
    ax_537.legend()
    # 第二个线性变换的BP网络
    f_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    y_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    # y'''(150,1)->f(150,1)
    f_apos_apos = np.exp(y_apos_apos_apos_target)
    y_apos_apos = f_apos_apos - np.cos(alpha_pred * np.pi / 2.0)
    f_apos = np.square(y_apos_apos)
    y_apos = f_apos + 1 - np.square(np.cos(alpha_pred * np.pi / 2))
    f = np.float32(0.5 * np.log(y_apos))
    # f(150,1)->y(150,64)
    W2 = tf.Variable(tf.compat.v1.truncated_normal([1, 64], mean=beta_target, stddev=0.2, dtype='float32'))
    b2 = tf.Variable(tf.zeros([1, 64]), dtype='float32')
    Wx_plus_b2 = tf.matmul(f, W2) + b2
    y = Wx_plus_b2
    # y(150,64)->output2(150,1)
    W2_out = tf.Variable(tf.divide(tf.ones([64, 1]), 64))
    b2_out = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_bw2_out = tf.matmul(y, W2_out) + b2_out
    output_2 = Wx_plus_bw2_out

    loss2 = tf.reduce_mean(tf.square(y_target - output_2))

    optimizer2 = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss2)
    init2 = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init2)
        for _ in range(3000):
            sess.run(optimizer2,
                     feed_dict={f_train: f_target, y_train: y_target})
            if _ % 1000 == 0:
                print('loss2', sess.run(loss2, feed_dict={f_train: f_target,
                                                          y_train: y_target}))
        y_pred = sess.run(output_2, feed_dict={f_train: f_target})
        W2_pred = sess.run(W2, feed_dict={f_train: f_target})
        b2_pred = sess.run(b2, feed_dict={f_train: f_target})
        W2_pred_mean = tf.reduce_mean(W2_pred).eval()
        b2_pred_mean = tf.reduce_mean(b2_pred).eval()
        beta_pred = W2_pred_mean
    ax_537.plot(f_target, y_pred, 'r-', label='y(f)')
    ax_538 = fig.add_subplot(5, 3, 8)
    ax_539 = fig.add_subplot(5, 3, 9)
    ax_538.hist(W2_pred[0], rwidth=0.9, label='W2_pred_mean=' + str(W2_pred_mean))
    ax_539.hist(b2_pred[0], rwidth=0.9, label='b2_pred_mean=' + str(b2_pred_mean))
    ax_538.legend()
    ax_539.legend()
    ax_537.legend()
    print('第二组神经网络拟合完成，权值W2{}，偏置b2：{}，换算得到系数beta：{}'.format(W2_pred_mean, b2_pred_mean, beta_pred))

    # 第三个BP的目标
    m_target = 1 / np.exp(y_pred)
    epsilon_target = delta_epsilon_target * m_target + epsilon_inf_target
    ax_5310 = fig.add_subplot(5, 3, 10)
    ax_5310.scatter(m_target, epsilon_target, label='epsilon(m)_T')
    ax_5310.legend()
    # 第三个线性变换的BP网络
    m_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    epsilon_train = tf.compat.v1.placeholder(tf.float32, [None, 1])
    # y(150,1)->m(150,1)
    m = np.float32(1 / np.exp(y_pred))
    # m(150,1)->epsilon(150,64)
    W3 = tf.Variable(tf.compat.v1.truncated_normal([1, 64], mean=delta_epsilon_target, stddev=100, dtype='float32'))
    b3 = tf.Variable(tf.compat.v1.truncated_normal([1, 64], mean=epsilon_inf_target, stddev=50, dtype='float32'))
    Wx_plus_b3 = tf.matmul(m, W3) + b3
    epsilon = Wx_plus_b3
    # epsilon(150,64)->output3(150,1)
    W3_out = tf.Variable(tf.divide(tf.ones([64, 1]), 64))
    b3_out = tf.Variable(tf.zeros([1, 1]))
    Wx_plus_bw3_out = tf.matmul(epsilon, W3_out) + b3_out
    output_3 = Wx_plus_bw3_out

    loss3 = tf.reduce_mean(tf.square(epsilon_target - output_3))

    optimizer3 = tf.compat.v1.train.GradientDescentOptimizer(0.0000001).minimize(loss3)
    init3 = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init3)
        for _ in range(3000):
            sess.run(optimizer3,
                     feed_dict={m_train: m_target, epsilon_train: epsilon_target})
            if _ % 1000 == 0:
                print('loss3', sess.run(loss3, feed_dict={m_train: m_target,
                                                          epsilon_train: epsilon_target}))
        epsilon_pred = sess.run(output_3, feed_dict={m_train: m_target})
        W3_pred = sess.run(W3, feed_dict={m_train: m_target})
        b3_pred = sess.run(b3, feed_dict={m_train: m_target})
        W3_pred_mean = tf.reduce_mean(W3_pred).eval()
        b3_pred_mean = tf.reduce_mean(b3_pred).eval()
        delta_epsilon_pred = W3_pred_mean
        epsilon_inf_pred = b3_pred_mean
    ax_5310.plot(m_target, epsilon_pred, 'r-', label='epsilon(m)')
    ax_5311 = fig.add_subplot(5, 3, 11)
    ax_5312 = fig.add_subplot(5, 3, 12)
    ax_5311.hist(W3_pred[0], rwidth=0.9, label='W3_pred_mean=' + str(W3_pred_mean))
    ax_5312.hist(b3_pred[0], rwidth=0.9, label='b3_pred_mean=' + str(b3_pred_mean))
    ax_5311.legend()
    ax_5312.legend()
    ax_5310.legend()
    print('第三组神经网络拟合完成，权值W3{}，偏置b3：{}，换算得到系数介电强度：{}，高频介电：{}'.format(W3_pred_mean, b3_pred_mean, delta_epsilon_pred,
                                                                    epsilon_inf_pred))
    ax_531.plot(freq_test, epsilon_pred, 'r-', label='epsilon(freq)')
    ax_531.legend()
    # TODO：这里需要计算最终的epsilon_final并且画出来
    # START:计算final值
    print(
        '开始计算最终拟合结果，预测参数如下'
        '\n\t介电不对称系数alpha:{}'
        '\n\t介电扩宽系数beta:{}'
        '\n\t弛豫时间tau:{}'
        '\n\t介电强度delta_epsilon:{}'
        '\n\t高频介电epsilon_inf:{}'.format(
            alpha_pred, beta_pred, tau_pred, delta_epsilon_pred, epsilon_inf_pred))
    # v1 = (omega*tau)^alpha
    v1 = np.power(np.multiply(omega_test, tau_pred), alpha_pred)
    # print('v1', v1)
    # p1 = alpha*pi/2
    p1 = alpha_pred * np.pi / 2
    # print('p1', p1)
    # v2 = [v1*sin(p1)]/[1+v1*cos(p1)]
    v2 = np.divide(np.multiply(v1, np.sin(p1)), 1 + np.multiply(v1, np.cos(p1)))
    # print('v2', v2)
    # phi = arctan(v2)
    phi = np.arctan(v2)
    # print('phi', phi)
    # B = cos(beta*phi)
    B = np.cos(np.multiply(beta_pred, phi))
    # print('B', B)
    # v3 = 1+2*v1*cos(p1)+v1^2
    v3 = 1.0 + np.multiply(v1, 2.0 * np.cos(p1)) + np.square(v1)
    # print('v3', v3)
    # A = (v3)^(-beta/2)
    A = np.power(v3, -beta_pred / 2.0)
    # print('A', A)
    # epsilon = epsilon_inf + delta_epsilon*A*B
    epsilon_final = epsilon_inf_pred + np.multiply(np.multiply(A, B), delta_epsilon_pred)
    ax_531.plot(freq_test, epsilon_final, 'g-', label='epsilon(freq)_FINAL')
    ax_531.legend()
    # END
    total_loss = np.sqrt(np.mean(np.square(epsilon_target - epsilon_pred)))
    print('预测完成,总loss：{}'.format(total_loss))


def main2(dc_bias):
    data_loader = DataLoader()
    freq_raw, epsilon_raw = np.array(data_loader.get_data(dc_bias), dtype='float32')
    omega_raw = np.multiply(freq_raw, 2.0 * np.pi)
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.scatter(freq_raw, epsilon_raw, label='epsilon(freq)_Raw')
    ax.legend()

    alpha_est = 0.95
    beta_est = 0.85
    tau_est = 1e-6
    epsilon_inf_est = np.min(epsilon_raw)
    delta_epsilon_est = np.max(epsilon_raw) - epsilon_inf_est
    popt, pcov = curve_fit(func, freq_raw, epsilon_raw
                           , p0=[alpha_est, beta_est, tau_est, epsilon_inf_est, delta_epsilon_est]
                           , bounds=([0, 0, 1e-8, 0, 0], [1, 1, 1e-5, 50, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    print('结果参数popt', popt)
    print('参数的协方差pcov', pcov)
    print('参数的标准差perr', perr)

    epsilon_pred = func(freq_raw, popt[0], popt[1], popt[2], popt[3], popt[4])
    ax.plot(freq_raw, epsilon_pred, 'r-', label='epsilon(freq)_Pred')
    ax.legend()


def func(freq, alpha, beta, tau, epsilon_inf, delta_epsilon):
    omega = np.multiply(freq, 2.0 * np.pi)
    # v1 = (omega*tau)^alpha
    v1 = np.power(np.multiply(omega, tau), alpha)
    # print('v1', v1)
    # p1 = alpha*pi/2
    p1 = alpha * np.pi / 2
    # print('p1', p1)
    # v2 = [v1*sin(p1)]/[1+v1*cos(p1)]
    v2 = np.divide(np.multiply(v1, np.sin(p1)), 1 + np.multiply(v1, np.cos(p1)))
    # print('v2', v2)
    # phi = arctan(v2)
    phi = np.arctan(v2)
    # print('phi', phi)
    # B = cos(beta*phi)
    B = np.cos(np.multiply(beta, phi))
    # print('B', B)
    # v3 = 1+2*v1*cos(p1)+v1^2
    v3 = 1.0 + np.multiply(v1, 2.0 * np.cos(p1)) + np.square(v1)
    # print('v3', v3)
    # A = (v3)^(-beta/2)
    A = np.power(v3, -beta / 2.0)
    # print('A', A)
    # epsilon = epsilon_inf + delta_epsilon*A*B
    epsilon = epsilon_inf + np.multiply(np.multiply(A, B), delta_epsilon)
    return epsilon


if __name__ == '__main__':
    # main是BP神经网络
    # main(0.0)
    # main2是scipy
    main2(0.0)
