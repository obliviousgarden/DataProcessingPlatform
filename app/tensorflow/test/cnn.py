import tensorflow as tf
import numpy as np


# 1 数据
class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000个, 28长, 28宽, 1色彩通道]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000个, 28长, 28宽, 1色彩通道]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


# 2 模型
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 卷积层1 (32卷积核，5*5感受野，，ReLU激励)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same） 64*64矩阵的5*5感受野1步长卷积结果是60*60矩阵，周围少的2圈数据用0补足(same模式)
            activation=tf.nn.relu   # 激活函数
        )
        # 池化层1(2*2窗口)
        # 对图像进行降采样64*64矩阵的2*2感受野2步长的池化结果是32*32
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # 卷积层2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        # 池化层2
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        # 数据压平层
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        # 全连接层1(尺寸1024，非线性，激励ReLU)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        # 全连接层2(尺寸10，线性)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # INPUT [batch_size, 28, 28, 1]
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32] 32个卷积核
        x = self.pool1(x)                       # [batch_size, 14, 14, 32] 2*2窗口池化
        x = self.conv2(x)                       # [batch_size, 14, 14, 64] 64个卷积核
        x = self.pool2(x)                       # [batch_size, 7, 7, 64] 2*2窗口池化
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64] 数据压平
        x = self.dense1(x)                      # [batch_size, 1024] 全连接1024
        x = self.dense2(x)                      # [batch_size, 10] 全连接10
        output = tf.nn.softmax(x)
        return output


# 3 训练
# 定义模型超参数
# 这里参数的意思是，所有样本需要训练5次，1次迭代中的样本数量是50，
# 假设样本总量是110个，那么样本就被划分成 110//50 = 2个样本batch，每1个batch都对应一次迭代，
# 那么遍历1次所有数据需要2次迭代，遍历5次所有样品需要 2*5 = 10次迭代
# (总迭代次数 = 样本总量train_data.shape[0] // 1次迭代样本量batch_size * 所有样本遍历波数num_epochs)
num_epochs = 5  # 1个epoch表示过了 1遍训练集中的所有样本
batch_size = 50  # 1次迭代中的样本数量
learning_rate = 0.001  # 学习速率

# 实例化多层模型，数据载入器，并且定义优化器
model = CNN()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# 训练过程
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    # 1 从DataLoader中随机取一批训练数据
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        # 2 将这批数据送入模型，计算出模型的预测值；
        y_pred = model(X)
        # 3 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用tf.keras.losses中的交叉熵函数作为损失函数；
        # 交叉熵函数多用于逻辑分类问题，线性回归需要选择其他函数
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    # 4 计算损失函数关于模型变量的导数；
    grads = tape.gradient(loss, model.variables)
    # 5 将求出的导数值传入优化器，使用优化器的apply_gradients方法更新模型参数以最小化损失函数。
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 评估
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 评估性能（预测正确占总样本的比例）
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size  # 开始的样本index和末尾的样本index
    y_pred = model.predict(data_loader.test_data[start_index: end_index])  # 使用这部分测试样本 基于上面的模型预测结果y_pred
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)  # 对比预测值和实际的图片
print("test accuracy: %f" % sparse_categorical_accuracy.result())
