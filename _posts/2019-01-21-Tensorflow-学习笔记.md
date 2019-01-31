---
layout:     post
title:      Tensorflow-学习笔记
subtitle:   学习mooc tensorflow实战开发的笔记
date:       2019-01-21
author:     IdiotLeo
header-img: img/beach_real_2.jpg
catalog: true
tags:
    - Deep Learning
    - Tensorflow
---

# Overview


这是IdiotLeo在学习慕课网基于Python玩转人工智能最火框架 TensorFlow应用实践时的笔记

# Tensorflow的原理以及进阶


数据模型--Tensor（张量）   计算模型--Graph（图）   运行模型--Session（会话）

##### Tensor（与numpy里Ndarray很像）


**张量的维度：**0--Scalar 标量； 1--Vector 向量； 2--Matrix 矩阵； 3--3Darray 3维向量；

**tensor的属性：**数据类型dtype（tf.float32\tf.int8\tf.string等），形状shape，其他（官网看）

**几种tensor：**

Constant（常量）--值不能改变的一种tensor，tf.constant（value,dtype=None,shape=None,verify_shape=False）

Variable(变量) --值可以改变的一种tensor，tf.Variable(...)  //必须至少传入输入值 eg:tf.Variable(4,dtype=tf.float64)

Placeholder(占位符) --先占住一个固定的位置，等着你之后往里面添加值的一种Tensor，tf.placeholder(dtype,shape=None,name=None)，在后续的程序里面可以给他赋值，赋值的机制是字典。feed_dict={ : }这样 

SparseTensor（稀疏矩阵） --一种稀疏的tensor，tf.SparseTensor()

**Tensor表示法：**

Tensor（"Mul:0",shape=(),dtype=float32） //Mul:0是名字

eg:<tf.Variable 'Variable_3:0' shape=() dtype=int64_ref>

##### 图Graph和会话Session


节点进行操作，节点间的线是tensor流。

sess=tf.Session()，会话。会话运行图。作用：让静态的图动起来。

Tensorflow程序流程：1、定义**计算图**结构；2、使用**会话**执行计算。

**重要理解：定义出来的各种组件用sess.run()来让他跑起来**

为什么没定义graph就可以用sess.run来跑呢？因为有默认的图

两种方法创建和关闭Session：一种直接建立sess，第二种用with tf.Session() as sess:(第二种方法不需要像第一种一样需要显式的关闭sess.close()这样)

# 可视化利器 TensorBoard及扩展工具

##### 1、用tensorflow保存图的信息到日志中


tf.summary.FileWritter("日志保存路径",sess.graph)

summary（总结、概览）  用于导出关于模型的精简信息的方法，可以使用tensorboard等工具访问这些信息

name_scope（命名空间）  类似namespace，包含/嵌套的关系，其含义主要为：

![tensorBoard解释](https://s2.ax1x.com/2019/01/28/kKXUG4.png)

##### 2、用rensorBoard读取并展示日志


tensorboard --logdir=日志所在地址

##### PlayGround

由js编写的网页应用，可以通过浏览器训练网络

##### 常用python库 Matplotlib

Matrix Plot Library   一个及其强大的绘图库

# 动手实现神经网络

##### 小练习：梯度下降解决线性回归

```
# -*- coding:UTF-8 -*-
用梯度下降的优化方法来快速解决线性回归问题

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构建数据
points_num = 100
vectors = []

#用Numpy的正态随机分布函数生成100个点，用这些点的（x,y）坐标值对应线性方程y=0.1*x +0.2，权重(Weight)是0.1，偏差(Bias)是0.2
for i in xrange(points_num):
	x1 = np.random.normal(0.0, 0.66)
	y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
	vector.append( [ x1, y1] )

x_data = [ v[0] for v in vectors ]   #真实点的x坐标
y_data = [ v[1] for v in vectors ]   #真实点的y坐标

# 图像1：展示100个随机数据点
plt.plot(x_data, y_data, 'r*', label="Original data")  #红色星形的点
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1] , -1.0, 1.0))  #初始化权重
b = tf.Variable(tf.zeros([1]))  #初始化偏差
y = W * x_data + b    #模型计算出来的y

# 定义loss function（损失函数）或cost function (代价函数)
# 对tensor的所有维度计算（(y - y_data)^2）之和 / N
loss = tf.reduce_mean(tf.square(y-y_data))

# 用梯度下降的优化器来优化loss
optimizer = tf.train.GradientDescentOptimizer(0.5)  #设置学习率0.5
train = optimizer.minimize(loss)

# 构建会话
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.gloabal_variables_initializer()
sess.run(init)

# 训练20步
for step in xrange(20):
	# 优化每一步
	sess.run(train)
	# 打印出每一步的损失，权重和偏差
	print("Step=%d, Loss=%f, [Weight=%f Bias=%f]")  \ %(step, sess.run(loss), sess.run(W), sess.run(b))

# 图像2：绘制所有的点并且绘制出得到的最佳拟合曲线
plt.plot(x_data, y_data, 'r*', label="Original data")  #红色星形的点
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line")  #拟合的线
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 关闭会话
sess.close()
```

##### 激活函数

作用：加入非线性。主要为tf.nn中activate function,tf.nn.relu(sofmax)等等。

```
# 激活函数的原始实现
import numpy as np
import matploitlib.pylot as plt
import tensorflow as tf

# 创建输入数据
x = np.linspace(-7, 7, 180) #（-7，7）之间的的等间隔180个点

# 激活函数的实现
def sigmoid(imputs):
	y = [1 / float(1 + np.exp(-x))for x in inputs]
	return y

def relu(inputs):
	y = [x * (x > 0) for x in inputs]
	return y

def tanh(inputs):
	y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
	return y

def softplus(inputs):
	y = [np.log(1 + np.exp(x)) for x in inputs]
	return y
	
# 经过tensorflow的激活函数处理的各个y值
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)

# 显示图像
plt.show()

# 创建会话
sess = tf.Session()

# 运行
y_sigmoid, y_relu, y_tanh = sess.run([y_sigmoid, y_relu, y_tanh])

# 创建各个激活函数的图像
plt.subplot(221)
plt.plot(x, y_sigmoid, c="red", label="Sigmoid")
plt.ylim(-0.2, 1.2)  # y的取值区间
plt.legend(loc="best")

# 关闭会话
sess.close()
```

##### 动手实现CNN卷积神经网络

MNIST：手写数据数据集

实现mnist的网络结构为：
![mnist的网络结构](https://s2.ax1x.com/2019/01/28/kKOScn.png)

```
import numpy as np
import tensorflow as tf

# 下载并载入mnist手写数字库（55000张 * 28 * 28）
from tessorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# one_hot ：独热码的编码（encoding）形式
# 0, 1, 2, 3, 4, 5, 6, 7, 8 9 的十位数字
# 0： 1000000000
# 1： 0100000000
# 2： 0010000000
# 以此类推

# None表示张量（Tensor）的第一个维度可以是任何长度
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
output_y = tf.placeholder(tf.int32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 从 Test 数据集里选区3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000] # 图片
test_y = mnist.test.labels[:3000] # 标签

# 构建卷积神经网络
# 第一层卷积，输出形状为[28*28*32]
conv1 = tf.layers.conv2d(
	inputs=input_x_images,  # 形状28*28*1
	filters=32,             # 32个过滤器（卷积核），输出的深度是32
	kernel_size=[5, 5],     # 过滤器在二维的大小是（5*5）
	strides=1,              # 不长是1
	padding='same'          # same表示输出的大小不变，也就是说要在外围补零两圈
	activation=tf.nn.relu   # 激活函数是relu
	)
# 第一层池化（亚采样），输出形状为[14, 14,32]
pool1 = tf.layers.max_pooling2d(
	inputs=conv1,           # 形状 [28*28*32]
	pool_size=[2, 2],       # 过滤器在二维的大小是（2 * 2）
	trides=2                # 步长是2
	)
	
# 第二层卷积，输出形状为[14*14*64]
conv2 = tf.layers.conv2d(
	inputs=pool1,           # 形状28*28*32
	filters=64,             # 64个过滤器（卷积核），输出的深度是64
	kernel_size=[5, 5],     # 过滤器在二维的大小是（5*5）
	strides=1,              # 不长是1
	padding='same'          # same表示输出的大小不变，也就是说要在外围补零两圈
	activation=tf.nn.relu   # 激活函数是relu
	)
# 第二层池化（亚采样），输出形状为[7, 7, 64]
pool1 = tf.layers.max_pooling2d(
	inputs=conv2,           # 形状 [14*14*64]
	pool_size=[2, 2],       # 过滤器在二维的大小是（2 * 2）
	trides=2                # 步长是2
	)
	
# 平坦化（flat）
flat = tf.reshape(pool2, [-1, 7, 7, 64]) # -1表示他根据确定的参数推断这个维度的大小,形状[7*7*64]

# 1024个神经元的全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout：丢弃50%,rate=0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 10个神经元的全连接层，这里不用激活函数来做非线性化了
logits = tf.layers.dense(inputs=dropout, units=10)  # 输出。形状[1, 1, 10]

# 计算误差（计算cross entropy（交叉熵），再用Softmax计算百分比概率）
# Softmax特点：输入几个，输出几个，总和为1
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# 用Adam优化器来最小化误差loss，学习率0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 精度。计算 预测值 和 实际标签 的匹配程度
# 返回(accuracy, update_op),会创建两个局部变量
accuracy = tf.metrics.accuracy(
	labels=tf.argmax(output_y, axis=1),
	prediction=tf.argmax(logits, axis=1， )[1]
	)
	
# 创建会话
sess = tf.Session()
# 初始化变量：全局和局部
init = tf.group(tf.global_variable_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
	batch = mnist.train,next_batch(50)  # 从头Train数据集里取下一个50个样本
	train_loss, train_op = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
	if i % 100 == 0:
		test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
		print("Step=%d, Train loss=%.4f, [Test_accuracy=%.2f]") % (i, train_loss, test_accuracy)

# 测试：打印20个预测值和真实值的对
test_output = sess.run(logits, {imput_x: text_x[:20]})
inferneced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers') # 推测的数字
print(np.argmax(test_y[:20], 1)), 'Real numbers')  # 真实的数字
```

##### 动手实现RNN-LSTM循环神经网络

CNN灵感:人类**视觉**皮层对外界事物的感知。RNN：人类的**记忆**机制。

RNN的优势：1、RNN的每一个输出与前面的输出建立起关联；2、能够很好处理序列化的数据（如音乐、文章等）；3、能够以前面的序列化对象为基础，来“生成”新的序列化对象。

RNN的局限性：步数增多导致 梯度爆炸/消失。

**梯度消失/爆炸：**0.9^100 = 0.00002656139（梯度消失）   1.1^100 = 13780.36(梯度爆炸)

**梯度爆炸解决：Gradient Clipping（梯度裁剪）：**![](https://i.loli.net/2019/01/31/5c5259a3477dc.png)

梯度消失类似于 记忆消散

**LSTM(Long Short-Term Memory)** ：一种特殊的RNN。1997年首次提出.

![传统RNN和LSTM](https://i.loli.net/2019/01/31/5c5260adac3a7.png)
![](https://i.loli.net/2019/01/31/5c525b4ba3e8c.png)

在t时刻，LSTM的输入有3个：1、X<sub>t</sub>：当前时刻网络的输入；2、h<sub>t-1</sub>：前一时刻LSTM的输出；3、C<sub>t-1</sub>：前一时刻的单元状态。

在t时刻，LSTM的输出有2个：1、h<sub>t</sub>：当前时刻LSTM的输出；2、C<sub>t</sub>：当前时刻的单元状态。

LSTM的神经元的“三重门”机制![LSTM的神经元的“三重门”机制](https://i.loli.net/2019/01/31/5c525d07b8342.png)

有一篇解读LSTM的文章，google，colah。

LSTM“门机制就像水坝的阀门”：取值[0,1]。

三重门重要性：遗忘门 > 输入门 > 输出门

LSTM解决梯度消失的主要原理：1、对需要的记忆保持久一些；2、不需要的记忆选择遗忘。

LSTM很多变体：比如GRU。

**Word Embedding**

One_hot编码模式：数据量很大时效率很低。  词向量编码模式数据量很大时效率高。

词向量：类似Clustering（聚类）

Word Embedding的学习资料：1、tensorflow官网里有Embeddings和Vector Representation of Words；2、知乎

**接下来要搭建的网络模型：**
![](https://i.loli.net/2019/01/31/5c52606e46add.png)

**工具代码：**
```
# -*- coding: UTF-8 -*-

"""
实用方法
"""

import os
import sys
import argparse
import datetime
import collections

import numpy as np
import tensorflow as tf

"""
此例子中用到的数据是从 Tomas Mikolov 的网站取得的 PTB 数据集
PTB 文本数据集是语言模型学习中目前最广泛的数据集。
数据集中我们只需要利用 data 文件夹中的
ptb.test.txt，ptb.train.txt，ptb.valid.txt 三个数据文件
测试，训练，验证 数据集
这三个数据文件是已经经过预处理的，包含10000个不同的词语和语句结束标识符 <eos> 的

要获得此数据集，只需要用下面一行命令：
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

如果没有 wget 的话，就安装一下：
sudo apt install wget

解压下载下来的压缩文件：
tar xvf simple-examples.tgz

==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代

# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch

==== 超参数（Hyper parameter）====
init_scale : 权重参数（Weights）的初始取值跨度，一开始取小一些比较利于训练
learning_rate : 学习率，训练时初始为 1.0
num_layers : LSTM 层的数目（默认是 2）
num_steps : LSTM 展开的步（step）数，相当于每个批次输入单词的数目（默认是 35）
hidden_size : LSTM 层的神经元数目，也是词向量的维度（默认是 650）
max_lr_epoch : 用初始学习率训练的 Epoch 数目（默认是 10）
dropout : 在 Dropout 层的留存率（默认是 0.5）
lr_decay : 在过了 max_lr_epoch 之后每一个 Epoch 的学习率的衰减率，训练时初始为 0.93。让学习率逐渐衰减是提高训练效率的有效方法
batch_size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目
（batch_size 默认是 20。取比较小的 batch_size 更有利于 Stochastic Gradient Descent（随机梯度下降），防止被困在局部最小值）
"""

# 数据集的目录
data_path = "data"

# 保存训练所得的模型参数文件的目录
save_path = './save'

# 测试时读取模型参数文件的名称
load_file = "train-checkpoint-69"

parser = argparse.ArgumentParser()
# 数据集的目录
parser.add_argument('--data_path', type=str, default=data_path, help='The path of the data for training and testing')
# 测试时读取模型参数文件的名称
parser.add_argument('--load_file', type=str, default=load_file, help='The path of checkpoint file of model variables saved during training')
args = parser.parse_args()

# 如果是 Python3 版本
Py3 = sys.version_info[0] == 3


# 将文件根据句末分割符 <eos> 来分割
def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


# 构造从单词到唯一整数值的映射
# 后面的其他数的整数值按照它们在数据集里出现的次数多少来排序，出现较多的排前面
# 单词 the 出现频次最多，对应整数值是 0
# <unk> 表示 unknown（未知），第二多，整数值为 1
def build_vocab(filename):
    data = read_words(filename)

    # 用 Counter 统计单词出现的次数，为了之后按单词出现次数的多少来排序
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    # 单词到整数的映射
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# 将文件里的单词都替换成独一的整数
def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# 加载所有数据，读取所有单词，把其转成唯一对应的整数值
def load_data(data_path):
    # 确保包含所有数据集文件的 data_path 文件夹在所有 Python 文件
    # 的同级目录下。当然了，你也可以自定义文件夹名和路径
    if not os.path.exists(data_path):
        raise Exception("包含所有数据集文件的 {} 文件夹 不在此目录下，请添加".format(data_path))

    # 三个数据集的路径
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # 建立词汇表，将所有单词（word）转为唯一对应的整数值（id）
    word_to_id = build_vocab(train_path)

    # 训练，验证和测试数据
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    # 所有不重复单词的个数
    vocab_size = len(word_to_id)

    # 反转一个词汇表：为了之后从 整数 转为 单词
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(word_to_id)
    print("===================")
    print(vocab_size)
    print("===================")
    print(train_data[:10])
    print("===================")
    print(" ".join([id_to_word[x] for x in train_data[:10]]))
    print("===================")
    return train_data, valid_data, test_data, vocab_size, id_to_word


# 生成批次样本
def generate_batches(raw_data, batch_size, num_steps):
    # 将数据转为 Tensor 类型
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size

    # 将数据形状转为 [batch_size, batch_len]
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    # range_input_producer 可以用多线程异步的方式从数据集里提取数据
    # 用多线程可以加快训练，因为 feed_dict 的赋值方式效率不高
    # shuffle 为 False 表示不打乱数据而按照队列先进先出的方式提取数据
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    # 假设一句话是这样： “我爱我的祖国和人民”
    # 那么，如果 x 是类似这样： “我爱我的祖国”
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    # y 就是类似这样（正好是 x 的时间步长 + 1）： “爱我的祖国和”
    # 因为我们的模型就是要预测一句话中每一个单词的下一个单词
    # 当然这边的例子很简单，实际的数据不止一个维度
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])

    return x, y


# 输入数据
class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # input_data 是输入，targets 是期望的输出
        self.input_data, self.targets = generate_batches(data, batch_size, num_steps)

```

**train.py**
```
# -*- coding: UTF-8 -*-

"""
训练神经网络模型

大家之后可以加上各种的 name_scope（命名空间）
用 TensorBoard 来可视化

from utils import *
from network import *

def train(train_data, vocab_size, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    # 训练的输入
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)

    # 创建训练的模型
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocab_size, num_layers=num_layers)

    # 初始化变量的操作
    init_op = tf.global_variables_initializer()

    # 初始的学习率（learning rate）的衰减率
    orig_decay = lr_decay

    with tf.Session() as sess:
        sess.run(init_op)  # 初始化所有变量

        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 为了用 Saver 来保存模型的变量
        saver = tf.train.Saver() # max_to_keep 默认是 5, 只保存最近的 5 个模型参数文件

        # 开始 Epoch 的训练
        for epoch in range(num_epochs):
            # 只有 Epoch 数大于 max_lr_epoch（设置为 10）后，才会使学习率衰减
            # 也就是说前 10 个 Epoch 的学习率一直是 1, 之后每个 Epoch 学习率都会衰减
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0)
            m.assign_lr(sess, learning_rate * new_lr_decay)

            # 当前的状态
            # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
            # 一个是 前一时刻 LSTM 的输出 h(t-1)
            # 一个是 前一时刻的单元状态 C(t-1)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))

            # 获取当前时间，以便打印日志时用
            curr_time = datetime.datetime.now()

            for step in range(training_input.epoch_size):
                # train_op 操作：计算被修剪（clipping）过的梯度，并最小化 cost（误差）
                # state 操作：返回时间维度上展开的最后 LSTM 单元的输出（C(t) 和 h(t)），作为下一个 Batch 的输入状态
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state], feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((datetime.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = datetime.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                    # 每 print_iter（默认是 50）打印当下的 Cost（误差/损失）和 Accuracy（精度）
                    print("Epoch {}, 第 {} 步, 损失: {:.3f}, 精度: {:.3f}, 每步所用秒数: {:.2f}".format(epoch, step, cost, acc, seconds))

            # 保存一个模型的变量的 checkpoint 文件
            saver.save(sess, save_path + '/' + model_save_name, global_step=epoch)
        # 对模型做一次总的保存
        saver.save(sess, save_path + '/' + model_save_name + '-final')

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    train(train_data, vocab_size, num_layers=2, num_epochs=70, batch_size=20,
          model_save_name='train-checkpoint')
```

**test.py**
```
# -*- coding: UTF-8 -*-

"""
测试神经网络模型

大家之后可以加上各种的 name_scope（命名空间）
用 TensorBoard 来可视化

from utils import *
from network import *


def test(model_path, test_data, vocab_size, id_to_word):
    # 测试的输入
    test_input = Input(batch_size=20, num_steps=35, data=test_data)

    # 创建测试的模型，基本的超参数需要和训练时用的一致，例如：
    # hidden_size，num_steps，num_layers，vocab_size，batch_size 等等
    # 因为我们要载入训练时保存的参数的文件，如果超参数不匹配 TensorFlow 会报错
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, num_layers=2)

    # 为了用 Saver 来恢复训练时生成的模型的变量
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 当前的状态
        # 第二维是 2 是因为测试时指定只有 2 层 LSTM
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
        # 一个是 前一时刻 LSTM 的输出 h(t-1)
        # 一个是 前一时刻的单元状态 C(t-1)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        # 恢复被训练的模型的变量
        saver.restore(sess, model_path)

        # 测试 30 个批次
        num_acc_batches = 30

        # 打印预测单词和实际单词的批次数
        check_batch_idx = 25

        # 超过 5 个批次才开始累加精度
        acc_check_thresh = 5

        # 初始精度的和，用于之后算平均精度
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                pred_words = [id_to_word[x] for x in pred[:m.num_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print("\n实际的单词:")
                print(" ".join(true_words))  # 真实的单词
                print("预测的单词:")
                print(" ".join(pred_words))  # 预测的单词
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc

        # 打印平均精度
        print("平均精度: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    if args.load_file:
        load_file = args.load_file
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    trained_model = save_path + "/" + load_file

    test(trained_model, test_data, vocab_size, id_to_word)
```
