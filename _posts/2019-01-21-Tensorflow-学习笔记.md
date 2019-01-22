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

# 可视化利器 TensorBoard

##### 1、用tensorflow保存图的信息到日志中

tf.summary.FileWritter("日志保存路径",sess.graph)

summary（总结、概览）  用于导出关于模型的精简信息的方法，可以使用tensorboard等工具访问这些信息

name_scope（命名空间）  类似namespace，包含/嵌套的关系，其含义主要为：

![tensorBoard解释](/img/tensorboard.png)

##### 2、用rensorBoard读取并展示日志

tensorboard --logdir=日志所在地址
