---
title: "Useful matlab code snippet"
date: 2018-08-30 00:00:00 +0800
categories:
  - Beautiful Code Fragment
layout: post
mathjax: true
---
# First one

Matlab里面应该多用向量运算，把循环语句转变为向量运算会省很多时间，程序也更简洁易读。

比方说，一个名叫array数组里面，你要将里面大于1的都变成0，就不必用到循环：

array(array>1)=0;

把大于1小于3的变成0。

array(and(array>1,array<3))=0;

当然，还可以使用find，这个也很好用。

# 挂后台防断网运行 run matlab from terminal

nohup matlab -r -nodisplay -nojvm scrip.m &;
…

# 用matlab还只会简单操作矩阵？？？那你也太low咯，让我来教教你

日期时间类型如果掌握了还是很有用的，比如，至少有些基础的图，你需要日期作为一个坐标轴吧，[datestr](https://www.mathworks.com/help/matlab/ref/datestr.html),
[datetime](https://www.mathworks.com/help/matlab/ref/datetime.html)了解一下

# 知道lambda表达式吗，是不是很强大，matlab中也有类似的东西哦，貌似叫啥[匿名函数](https://ww2.mathworks.cn/help/matlab/matlab_prog/anonymous-functions.html)

MATLAB通过定义匿名函数来增强符号运算的功能，哦，原来与数值运算相对的符号运算其实是函数式编程。定义匿名函数的格式为：

```
@(<parameters>) <body>
例如：
sum_xy = @(x, y) x + y
```

但是匿名函数作用可不仅仅局限于此哟，
 他可以嵌套的，多重匿名函数，可以实现参数的传递等功能
 myfunction = @(x,y) (x^2 + y^2 + x*y);

```
x = 1;
y = 10;
z = myfunction(x,y)
output: z = 111
```

但是还没找到可以再一个匿名函数顺序执行语句的其他方法，有点小忧桑，mark一下

# 今天看了matlab的数据类型感觉挺有用的:table,cell,和最基本的matrix，哦对咯，别忘了struct呀

# 发现了一个可以让代码更简洁，减少for循环的方法：

```
1. 可以考虑用bsxfun, arrayfun,[戳这儿](https://www.mathworks.com/help/matlab/ref/arrayfun.html), cellfun, structfun等对矩阵中的每一个元素来进行同样的操作。

S(1).f1 = 1:10;
S(2).f1 = [2; 4; 6];
S(3).f1 = []

[nrows,ncols] = arrayfun(@(x) size(x.f1),S)
        nrows = 1×3

             1     3     0

        ncols = 1×3

            10     1     0
```

# matlab里面可以通过两个%%加一个空格对代码分段，并可以通过快捷键对每一段单独运行

# 还有通过python来调用matlab的引擎进行计算[戳我](https://blog.csdn.net/sunny_xsc1994/article/details/79254196)

# 其他小的tricks

{
 1 .m脚本文件选中对应行，按F9可以快速执行

1. .m脚本文件快速执行： 按F5
1. ctrl + i 可进行多行自动对齐
1. ctrl + [ 可进行多行左缩进
1. ctrl + ] 可进行多行右缩进
1. 按住shift，光标所在处，鼠标点击后，则选中点击区域与光标所在处中间的区域

---

作者：夜月xl
来源：CSDN
原文：[https://blog.csdn.net/u013045749/article/details/40429173](https://blog.csdn.net/u013045749/article/details/40429173)
版权声明：本文为博主原创文章，转载请附上博文链接！
}
