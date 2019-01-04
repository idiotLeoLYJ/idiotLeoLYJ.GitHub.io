---
layout:     post
title:      Tips for assembler
subtitle:   课程系统设计汇编器😅
date:       2019-01-03
author:     IdiotLeo
header-img: img/beach_real_1.jpg
catalog: true
tags:
    - Assembler
    - 汇编器
---

# Overview

课程设计汇编器部分的注意要点。

### SeuLex.sln

##### main()

**step 1、对词法分析文件asmlex.l进行提取（analysis）**

ifile.open(asmlex.l)

**analysis(is2reTable,actionTable)；**

（is2reTable以hash表形式存提取出的各种正则式，pair，类似 first:T_RS   second:$((v(0|1))|(a(0|1|2|3))|(t(0|1|2|3|4|5|6|7|8|9))|(s(0|1|2|3|4|5|6|7|8|9))|(i(0|1))|(sp)|(ra)|(zero)|(0|1|2|3|4|5|6|7|8|9)|((1|2)(0|1|2|3|4|5|6|7|8|9))|(3(0|1)))）

(actionTable存action中的对应内容，里面寸的是对应该怎么做，例如{ return "T_DATA";})  有一个变量lineno记录条数

在这里面对asmlex自然有对定义段规则段的一些处理（对 % 的识别）

**compleRe(re,id2reTable);**  （目前没发现有什么大用）一个问题？？？？

目前这个函数是把{}这种用户自定义的规则提取出来写进is2reTable（把{T_WORDNUM}换成了对应的(-(0x(0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|a|b|c|d|e|f)*(0|1|2|3|4|5|6|7|8|9)*))|(0x(0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|a|b|c|d|e|f)*(0|1|2|3|4|5|6|7|8|9)*)）

int actionTableIndex=actionTable.size()-1;

Re[l]=re;（string Re[42];）

这个Re[]中就包含了所有的带.号分隔和带#结尾的正则表达式；

Analysis Done！！

**Step2、将正则表达式转换到NFA**

sub_nfa final;  final=re2nfa(Re[i]);

相关数据结构：

struct sub_nfa:1、edge  EdgeSet[MAX];2、unsigned int EdgeCount;3、unsigned int StartState;4、unsigned int EndState;

struct edge：1、unsigned int value;//边上标识符；unsigned int from;//边起始状态；3、unsigned int to;//边终止状态；

这里是对re进行转化到nfa，具体转化方法为利用之前的.号间隔，使用一个辅助栈status（用来记录当前状态的前驱后驱问题），对.* l几种符号进行不同的选择处理。将得到的nfa存储在nfa数据结构中。

final=re2nfa(Re[i]);这个是得到某一条re的各种状态以及转换状态，之后再在produceNfa中进行转化从开始一直到终态获得NFA；

这里面就涉及到一个对re转化nfa图的分支处理（？？）

produceNfa就是把得到的各状态情况从startstate一直串联到endstate，最后再加一个终结态。

### SeuAssembly.sln

**Step1、构建pda，生成lr1**

Item里面存的生成式

ap3里面，first存的序号，second寸的{}这种对应动作

terminal里存的是所有终结符，pair类型，first 是什么；second 序号。Noterminal也一样。

还不知道hpset是干什么的

produceActionTable.insert(ap3);这个是将动作存起来；每一项first为size；second为具体动作；

之后到translateAction（）中

**Step2、构建PDA**

yacc.createPDA();就是之前编译课设的那一部分。暂时跳过；

**Step3、生成LR（1）分析表，并输出到actionTable.txt和GotoTable.txt中**

yacc.createTable();

void SeuYacc::outputLR1Table(){  //将LR1分析表输出到tableYacc.h（ACtionTable和GotoTable）

void SeuYacc::outputAction(){  //将产生式对应的动作输出到actionYacc.h

**Step4、**


