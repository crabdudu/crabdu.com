---
title: "Neural Net的理解"
date: 2022-03-08T02:33:04-05:00
draft: false
tags: ["Meteorology","Python","DeepLearning"]
categories: ["Notes"]
keywords:
- Python
- DeepLearning
description: "深度学习的一些基本理解"
---
# Neural Net的理解
神经网络做的事情从计算机的角度来看：可以被简单的理解为Input矩阵与Weight矩阵和bias矩阵相互作用后，输出Output矩阵。   

上面的过程从**人**的角度来看： **输入特征**与**权重矩阵和偏置矩阵**相互作用后（作用方式取决于不同的模型），输出一些想要的东西。

而怎么获得一个能够满足要求的权重矩阵和偏置矩阵就是不同神经网络的区别。

举一个实例：Andrew在自家的后花园养了许多猪，可总有许多偷猪贼想要偷猪，不巧的是这些偷猪贼都在半夜行动，愤怒的Andrew在后花园设立了带有神经网络图像识别功能的狙击枪，那么显而易见，输入特征为狙击枪内所看到的图片（Input矩阵），经过神经网络作用（与Weight矩阵和bias矩阵相互作用）后得到了关于看到的东西是人是猪的判断（Output矩阵）

神经网络在很多年前就已经出现，不过当时无法解决多层神经网络中存在的梯度消失等一系列问题，所以输入特征都是由人工提取的。而在过去的几年，有人提出了一些方法，解决了一系列问题，提出构造一个具有非常多层的神经网络，让神经网络自己学习特征。于是渐渐有了深度学习的概念。

## 我们需要做的：
***
1. 处理输入特征，希望狙击枪内看到的图像足够干净清楚。
2. 训练神经网络，祈祷它足够聪明，不会把猪给打死却放跑偷猪贼。

## 实际过程中的通用步骤
***
1. 加载dataset
2. 搞清楚dataset的形状包括（m_train, m_test, features)
3. 特别地，在图像识别中，要将train_set_orig展平为train_set_flatten
4. 进行标准化，也就是压缩到(0,1)之间 
5. 构造sigmoid函数 
6. 构造初始化initialize_with_zeros函数 
7. 构造propagate函数 
8. 构造optimize函数 
9. 构造predict函数 
10. 使用上面的函数构造model函数 
11. 模型测试(optional)
12. 模型训练 
13. 个例查看 
14. 学习曲线绘制 
15. 用实际例子去测试

# 深度学习的神经网络构造流程
## 数据处理
***

1. 定义好全局变量
2. 首先用Numpy或者其他任何方式设置好需要训练的数据，无论是一维二维还是多维。
3. 然后统统用toch.tensor转为tensor数据，否则torch无法处理。当然直接用torch.函数直接构造需要的数据也是可以的。特别地，如果需要高精度的数据可以使用torch.Tensor来转为tensor数据。
此外，如果读入的数据是一维数据，那么需要用torch.unsqueeze(一维数据)来扩充纬度，因为torch并不处理一维数据。  
4. 有了tensor数据就可以用tensor类的各种方法，在这一步中仍然可以对数据处理，不过最好在这之前用Numpy就把所有数据处理好。
5. 然后用dataset将tensor数据封装起来
6. 然后用Dataloader再将dataset封装起来，同时要输入批训练一类的参数。


## 构造Net类
***

1. 先继承torch.nn.Module，初始化和声明神经网络层，选用合适的神经网络模型（CNN,RNN一类的）。
2. 构造输出
3. 定义forward方法，使用刚声明的神经网络层具体怎么处理数据
4. 返回输出结果

## 主函数
***
### 根据具体现实的数据构造相适应的体系
1. 构造实例net
2. 选择合适的修正函数optimizer
3. 选择合适的损失函数loss_function
### 构造训练循环
1. 训练多少个周期
2. 把需要预测的数据传入实例net中
3. 用损失函数算出预测值和真值之间的差别 
4. 修正函数将梯度清零optimizer.zero_grad()
5. 用损失值反向求导函数loss.backward() ，这两步应该就是学习的关键，不过深层次的理解的不好，看不清内部。 
6. 修正函数向前求导optimizer.step

## 可视化
***
plt相关库，使用matplotlib库进行可视化，此步骤的主要目的不一定是为了落地结果，可以简单画出测试数据以此来判断参数的设置是否欠拟合或者过拟合了。

## Q&A：
1. Q: 为什么不直接用Numpy的Xarray数据进行训练而要这么折腾转为tensor数据   
A: 因为numpy不会GPU加速计算，而是CPU加速运算，所以这也同时解释了为什么数据量越大numpy速度优势就会越明显，而DL需求的并不是高速的小数据吞吐，
而是需要大数据吞吐（当然高速度也需要，但鱼和熊掌不可兼得），能够让GPU加速承担该任务的Tensor数据自然就是最好的选择。当然还有另一个原因，tensorflow又不是numpy家的，两家人不说一家话，就像特斯拉和巨无霸卡车一样，都是车（库），但目的不同。

2. Q: 为什么不对dataset直接进行操作而要将dataset封装成dataloader？   
A: 由于批次训练的需求存在，直接对dataset操作会很麻烦，所以将其全部打包成dataloader，然后利用参数（shuffle，sample，batch_size等等）间接对dataset进行处理。这样既方便又高效。


# 神经网络（Artificial Neural Network/ANN）的分类


## 前馈神经网络(feedforward neural network/FNN)
***


## 卷积神经网络 (Convolutional Neural Network/CNN)
***


### 什么是卷积神经网络？卷积神经网络的定义
A : 卷积神经网络（Convolutional Neural Network, CNN）是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。

卷积神经网络由一个或多个卷积层和顶端的全连通层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在图像和语音识别方面能够给出更好的结果。这一模型也可以使用反向传播算法进行训练。相比较其他深度、前馈神经网络，卷积神经网络需要考量的参数更少，使之成为一种颇具吸引力的深度学习结构。
[Wiki](https://zh.wikipedia.org/zh-hk/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)


### 卷积相较于全连接神经网络区别于优势
A : 传统的三层神经网络（输入层、隐藏层、输出层）是一种将所有数据都读入训练，而这样的训练在面对有价值的样本数据时（即每一个数据都会产生很关键的作用）可以保持训练速度和效果，
但当训练的数据是图像此类的数据，其数据量不仅大很多，且有很多“无用数据”或者说一堆数据表现同一个特征，所以卷积神经网络应运而生，通过卷积核窗口的不断移动和池化提取出了一个能够代表区域的特征值


### 卷积核为什么有效
A : 卷积过程就是一个区域特征提取的过程，卷积核就是一个过滤器，通过各种各样的过滤器结合目标任务所需要的特征，不断的卷积操作便可以将图片提取出某些特征，便达到了分类的效果。例如，有一张图片被一个三维数组所表示，[length, width, color] 那么如果某个卷积核沿着第一纬度滑动卷积后在第三维其值一直为0，很容易就能判断出在垂直方向上有一条颜色一致的线。同理，
在第二纬卷积就可以知道在水平方向上有一根颜色一致的线，若第三维值变化剧烈，那说明在水平方向上图案不连贯。通过各种各样的特征提取，于是计算机就“认识了”图片。
### 为什么卷积核是正方形？卷积核可以是其他形状吗？
A : 通常来说，卷积核都是正方形，例如3 * 3或者5 * 5。其实具体选择什么形状的卷积核更加取决于最终的任务，
因为卷积的目的是提取特征，无论选择什么形状的卷积核服务的都是如何更好的提取特征，举个例子：在图片中要识别人的特征，高而窄的卷积核就更容易发现特征；要识别轮船，宽而矮的卷积核更容易提取特征。另外，也可能是因为正方形的卷积核更容易计算，毕竟规则矩阵的计算更快，也有更多方法可以使用。

## 长短期记忆网络(Long Short Term Memory Network/LSTM)
***
https://mofanpy.com/tutorials/machine-learning/torch/intro-RNN/

https://mofanpy.com/tutorials/machine-learning/torch/intro-LSTM/

# 深度学习和机器学习的区别联系

## 机器学习
***
### 什么是机器学习？机器学习的定义
A : An application of artificial intelligence that includes algorithms that parse data, 
learn from that data, 
and then apply what they’ve learned to make informed decisions. In summary **it performs a function with the data given to it and gets progressively better over time.**


## 深度学习
***
### 什么是深度学习？深度学习的定义
A : A subfield of machine learning that structures algorithms in layers to create an “artificial neural network” that can learn and make intelligent decisions on its own.
### 胡编乱造的理解
机器学习是指计算机会给出一个可解释的决策树或者决策模型用于判断资料，具有比较强的可解释性，是通过算法的不断进步、先验知识提供更好的特征而发展的的。

而深度学习从属于机器学习，是机器学习中的一个方法，它表明某个作用是通过给的资料不断学习变好的，强调其**自发**的存在，这个自发性其实看到代码上就是每层的神经元在每一个epoch当中不断的优化自己的权重，通过反复的学习利用各种数学法则给出了一套最适合于当前问题的权重参数。
但由于其隐藏层几乎没有物理意义无法对应至理论中，而且是特定问题特定学习，所以离开特定问题这套参数就毫无价值或者说大打折扣。  

本质上来说，深度学习学的是调参，参数调得好，优化器可以以最快的梯度下降速度找到最优解。

为何深度学习要求计算大量的简单任务，内核还是计算机在通过不断的修正参数使得残差函数收敛，而人不能在一瞬间试这么多次算这么多东西罢了。

[参考](https://www.zendesk.com/blog/machine-learning-and-deep-learning/)

# 深度学习应用
***
## 可以应用气象领域中的哪些方面
深度学习的特点决定了他需要大量的数据进行梯度下降的工作，又因为一般的结构化数据是更容易操作的，这两个特点正好与气象数据相符合。
### 气象研究的特点
众所周知，由于气象的实验条件具有不可复制性，从而气象更多的是由动力物理化学等角度出发的确实理论指导数值模式方程的建立，通过预报数据与观察数据之间的差异不断修正参数从而实现下一次更好的预报。
### 深度学习在气象中的地位
深度学习应该说是一种“数学游戏”，他没有一个可依托的物理解释，更像是在玩一个无论你用什么方法比谁先找到最优解的游戏。

在学习的过程中深刻的感受到了：“深度学习像炼丹”；“做AI的离不开参数”。

事实上，深度学习就像是一个炼丹炉，在吃下“丹”之前（训练好模型并进行实际数据验证），没有人知道这个模型对哪些数据有效、对什么时次的数据有效、对扰动量是否有效。
因为无论在神经网络的哪个阶段都只有数学模型在指导在修正，而不是融合了目标领域的一些已知的理论进行指导，最终的结果就是一套“行之有效”的参数+模型，但却没有合适的研究领域的理论去解释，是一个实践超前于理论的领域。

就在此刻，我更认为深度学习是一个工具，他可以做很多非线性化的探索、他可以从细枝末节的地方发现特点、他可以在了解机理之前就做出预测，但他不懂为什么...  


