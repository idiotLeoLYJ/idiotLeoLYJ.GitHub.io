---
layout:     post
title:      OS_Labs exps of ucore
subtitle:   基于Ucore操作系统实验😅
date:       2018-07-01
author:     IdiotLeo
header-img: img/beach_real_2.jpg
catalog: true
tags:
    - Operating System
    - Ucore
---

>Notes:本文仅为IdiotLeo学习ucore时遇到的关键问题的记录。

# OS_Ucore

THU OS Labs based on Ucore!

陈渝老师维护的github[网址](https://github.com/chyyuu/os_course_info)

详细的操作系统实验[指导手册](https://chyyuu.gitbooks.io/ucore_os_docs/content/)

### Lab 0 : 实验准备以及环境搭建


##### 了解OS实验


我们准备如何一步一步来实现ucore呢？根据一个操作系统的设计实现过程，我们可以有如下的实验步骤：

1、启动操作系统的bootloader，用于了解操作系统启动前的状态和要做的准备工作，了解运行操作系统的硬件支持，操作系统如何加载到内存中，理解两类中断--“外设中断”，“陷阱中断”等；

2、物理内存管理子系统，用于理解x86分段/分页模式，了解操作系统如何管理物理内存；

3、虚拟内存管理子系统，通过页表机制和换入换出（swap）机制，以及中断-“故障中断”、缺页故障处理等，实现基于页的内存替换算法；

4、内核线程子系统，用于了解如何创建相对与用户进程更加简单的内核态线程，如果对内核线程进行动态管理等；

5、用户进程管理子系统，用于了解用户态进程创建、执行、切换和结束的动态管理过程，了解在用户态通过系统调用得到内核态的内核服务的过程；

6、处理器调度子系统，用于理解操作系统的调度过程和调度算法；

7、同步互斥与进程间通信子系统，了解进程间如何进行信息交换和共享，并了解同步互斥的具体实现以及对系统性能的影响，研究死锁产生的原因，以及如何避免死锁；

8、文件系统，了解文件系统的具体实现，与进程管理等的关系，了解缓存对操作系统IO访问的性能改进，了解虚拟文件系统（VFS）、buffer cache和disk driver之间的关系。

其中每个开发步骤都是建立在上一个步骤之上的，就像搭积木，从一个一个小木块，最终搭出来一个小房子。在搭房子的过程中，完成从理解操作系统原理到实践操作系统设计与实现的探索过程。

![](http://ow7qvf5zp.bkt.clouddn.com/image001.png)


##### 实验环境

<strong>virtualbox</strong> + <strong>配置好的ubuntu x64</strong>[映像云盘地址](https://pan.baidu.com/s/11zjRK) 

下载代码

< $ git clone https://github.com/chyyuu/ucore_lab.git
