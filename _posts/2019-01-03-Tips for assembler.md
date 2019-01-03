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

step 1、对词法分析文件asmlex.l进行提取（seuLex）

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

Step2、将正则表达式转换到NFA

