import tensorflow as tf
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from django.core import serializers
import tensorflow as tf
import numpy as np
import cv2 as cv


@require_http_methods(["GET"])
def tf_tem(request):
    response = {}
    print(request.GET)
    # TODO：调取数据库信息对图片进行分析处理
    response['code'] = 738
    response['data'] = {'data': ''}
    response['msg'] = '分析成功'
    return JsonResponse(response)


if __name__ == '__main__':
    # 定义一个随机数（标量）
    random_float = tf.random.uniform(shape=())

    # 定义一个有2个元素的零向量
    zero_vector = tf.zeros(shape=(2))

    # 定义两个2×2的常量矩阵
    A = tf.constant([[1., 2.], [3., 4.]])
    B = tf.constant([[5., 6.], [7., 8.]])
    # 查看矩阵A的形状、类型和值
    print(A.__str__())

    # 计算矩阵A和B的和
    C = tf.add(A, B)
    # 计算矩阵A和B的乘积
    D = tf.matmul(A, B)

    print(C.__str__())
    print(D.__str__())

    x = tf.Variable(initial_value=3.)
    # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    with tf.GradientTape() as tape:
        y = tf.square(x)
    # 计算y关于x的导数
    y_grad = tape.gradient(y, x)
    print(y, y_grad)
