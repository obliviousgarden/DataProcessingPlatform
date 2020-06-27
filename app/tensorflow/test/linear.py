import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense是Keras下的全连接层，对输入矩阵call内部的input进行线性变换和激励
        # 这里没有激励函数，所以就是单纯的线性变换
        # Input(batch_size × input_dim) MATMUL Kernel(input_dim × units) + Bias(units) = Output((batch_size × units)
        self.dense = tf.keras.layers.Dense(
            # 输出张量Output的维度
            units=1,
            # 激活函数
            activation=None,
            # 权重矩阵，默认tf.glorot_uniform_initializer
            kernel_initializer=tf.zeros_initializer(),
            # 偏置向量，默认tf.glorot_uniform_initializer
            bias_initializer=tf.zeros_initializer()
        )
    # ！！！不要重载__call__函数
    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()  # 上方定义的线性模型
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) # 优化器SGD学习效率0.01
for i in range(100):
    with tf.GradientTape() as tape:  # 记录梯度上下文
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))  # 实际值和预测值之间的loss函数
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 利用优化器更新tf的变量
print(model.variables)
