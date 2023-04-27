---
title: "Pytorch入门"
date: 2022-01-06T01:50:13-05:00
draft: false
tags: ["Python","Pytorch","DeepLearning"]
categories: ["Notes"]
keywords:
- Python
- Pytorch
description: "pytorch入门内容"
---
# pytorch入门

## 基本内容
***
### tensor数据构造
torch可以被看作是pytorch中的numpy，区别在于numpy只能调用cpu，而pytorch可以调用GPU工作。
```python
import  torch
import  numpy as np
```
```python
def run():
    numpy_data	= np.arange(6).reshape((2,3))  
    np2torch	= torch.from_numpy(numpy_data) # xdarray转tensor
    torch2np	= torch_data.numpy()

    data = [-1 ,-2 ,-3]
    tensor = torch.FloatTensor(data) # 列表直转tensor
    
    print(
    	'\nnumpy_data:', numpy_data,
    	'\np2torch:', np2torch,
    	'\torch2np:', torch2array
    )
    
    print(
        '\np.abs', np.abs(data),
        '\tensor.abs', torch.abs(tensor)
    )
    
    return
```
*Tips：tensor数据不像是np的xarray数据一样可以直接对列表一类的数据进行直接操作，需要先将其转换为tensor类型，在tensor前加特定类型名可以声明特别的类型。* 
例如`tensor = torch.FloatTensor(data)`


### 矩阵相乘
***
```python
def matrix():
    data = [[1, 2],[3,4]]
    tensor  = torch.tensor(data)

    print(
        '\nnumpy的矩阵相乘:',    np.matmul(data, data),
        '\ntorch的矩阵相乘：',   torch.mm(tensor, tensor),
    )
    return
```


### torch中的“变”量
***
```python
import torch
from torch.autograd import Variable

def run():
    data = [[2,3], [4,5]]
    tensor  =   torch.FloatTensor(data)
    variable = Variable(tensor, requires_grad = True)
    print(
        'tensor:',tensor,
        '\nvariable:', variable,
    )
    t_out = torch.mean(tensor*tensor)
    v_out = torch.mean(variable*variable)

    v_out.backward()
    print(t_out)
    print(variable.grad)

    print(variable)
    print(variable.data)
    print(variable.data.numpy())

    return
```
tensor可以被看作一个一个可以被torch操作的变量，而variable则可以被看作将tensor、tensor的梯度、得到tensor梯度的方式三个属性封装起来的一个动态变量  
variable的特点就是他不是一个静态的量，是一个动态变化的量。


### Regression
***
这个神经网络主要是用于线性回归
```python
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F
import matplotlib.pyplot as plt
```
F是激活函数，其目的是将线性问题通过处理转化为非线性问题，常见的激活函数有：
- ELU  
  数学表达式为：ELU(x)=max(0,x)+min(0,α∗(exp(x)−1))  
- RELU  
  数学表达式为ReLU(x)=max(0,x)，RELU可以将负值取0，将正值保留。

```python
def run():
    # creat fake data
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y) # 将x, y封装为变量，让其动态变化

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()
    # for enviewing original data
    
```
以下为如何构造一个线性回归。
```python
    class Net(torch.nn.Module):
        def __init__ (self, n_feature, n_hidden, n_output):
            super(Net, self).__init__() 
            self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
            self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
            # 首先要定义一个类去继承torch.nn.Module的模块
            # 然后初始化示例具有哪些属性
        def forward (self, x):
            x = F.relu(self.hidden(x))  # activation function for hidden layer
            x = self.predict(x)  # linear output
            return x
            # forward function 就是具体的方法，包括对数据进行什么样的操作（例如施加一个激活函数），如何输出等
    net = Net(n_feature=1, n_hidden=10, n_output=1)  # 构造一个具体的net
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2) # 用SGD修正函数，似乎Adam是更好更快的，看需求。
    loss_func = torch.nn.MSELoss()  # 损失函数的量化

    plt.ion()  # 实时打印
```
以上构造了一个含有隐藏层和输出层的神经网络。

接着进行训练
```python
    for t in range(200): # 训练两百次 
        prediction = net(x)  # input x and predict based on x

        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__=='__main__':
    run()
```
这就是一个典型的训练过程：

1. 处理资料（手段包括但不限于切片、插值、归一化等等），严格的来说，深度学习就是为数据资料服务的，更多的时间被应用于处理资料而不是网络的构建。
2. 构建神经网络，在这个过程中是连接输入与输出的重要步骤，明确的输出需求和足够详细的输入资料特征会更好的构建网络。（堆叠功能层、选择acf、opt、loss等）
3. 具体的训练过程，只是训练是很简单的，但为了更好的效果需要控制变量和早停法等，搭配tensorboard或者w&b等服务进行可视化是一个不错的选择。



### Classification分类问题
***

这个模型主要用于不同的分类
```python
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def run():
    # make fake data
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

    # The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
    # x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):  # standard procedure, put the parameters to the function
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, n_hidden) # start to build the network, define every layer what they want
            self.output = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):   # specifically, to defien what the layera are doing.
            x = F.relu(self.hidden(x))
            x = self.output(x)
            return(x)

    net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)      #optimize the loss
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    plt.ion()   # something about plotting

    for t in range(100):
        out = net(x)                 # input x and predict based on x
        loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if t % 2 == 0:
            # plot and show learning process
            plt.cla()
            prediction = torch.max(out, 1)[1]
            pred_y = prediction.data.numpy()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='viridis')
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'blue'})
            plt.pause(0.1)

    plt.ioff()

if __name__=='__main__':
    run()
```

## 批训练
***
在实际的训练过程中，由于训练样本的数量都很大，训练时间很长，通常需要使用批次训练，
即将样本分批次放入模型中重复训练，具体流程可看[DL_progress](/posts/DL_progress/DL_progress.md)，在某些情况下，例如yolo框架中，将批训练设置为1可能会有奇效。

```python
import torch
import torch.utils.data as Data

torch.manual_seed(1) #保证每次初始化一致

BATCH_SIZE = 5
#creat fake data
x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x ,y) # take the normal dataset convet to tensor dataset
# batch traininig, input the dataset, batch_size, shuffle and num_workers
loader = Data.DataLoader(
    dataset= torch_dataset,
    batch_size= BATCH_SIZE,
    shuffle= True,
    num_workers= 2,
)

def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())

if __name__ == '__main__':
    show_batch()
```

## 不同的修正函数收敛速度
***
```python
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

#hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

#creat fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
print(type(x), type(y))
plt.scatter(x.numpy(), y.numpy())
plt.show()

#Conter dataset to tensordataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden =  torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':
    net_SGD = Net(1, 20, 1)
    net_Momentum = Net(1, 20, 1)
    net_RMSprop = Net(1, 20, 1)
    net_Adam = Net(1, 20, 1)
    nets=[net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]   # record loss

    for epoch in range(EPOCH):
        print('Epoch:', epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(batch_x)  # get output for every net
                loss = loss_func(output, batch_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data.numpy())  # loss recoder


    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
```

## 卷积神经网络(CNN)
***
具体的理解可以看这篇[DL相关问题](/posts/DL_progress/DL&ML_ideas.md)中的CNN部分。
```python
import torch
import torch.nn as nn

class Cnn(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(                                          # (1纬高度因为是黑白, 28, 28)
                in_channels= 1, # height of input data
                out_channels= 16, # height of output data
                kernel_size= 5, # width and length of filter
                stride= 1, # lenth of step
                padding= 2, # 把原始图片边界加上的宽度，为了能够使输出的和输入的大小一致。 一般5*5的kerel_size对应2的padding，3*3的对应1，保证输入输出一致。
            ),                                                  # 由于有padding所以其卷积出来的长宽没变还是(1, 28, 28)
            nn.ReLU(),                                          # RELU不改变长宽(1, 28, 28)
            nn.MaxPool2d(kernel_size= 2,), # 池化减小卷积层的大小   # 池化没有padding，输出大小变为了 (1, 14, 14)
        )
        self.conv2 = nn.Sequential(                             # 同理(1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),                         # (1, 14, 14)
            nn.ReLU(),                                          # (1, 14, 14)
            nn.MaxPool2d(2)                                     # 池化使得输出变小(1, 7 ,7)
        )
        sel.out = nn.Linear(32 * 7 * 7,  10)                    # 输出层要展平

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = x.view(x.size(0), -1)                               # view() = reshape()
        output = self.out(x)
        return output

```
## RNN-LSTM神经网络
***
```python
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy


# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data

train_data = dsets.MNIST(
    root='./data',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(
    root='./data/',
    train=False,
    download= DOWNLOAD_MNIST,
    transform=transforms.ToTensor()
)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy()[:2000]    # covert to numpy array

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE, # 输入的像素点
            hidden_size=64, # 隐藏层的数量
            num_layers=1, # 越高越准，训练时间越长
            batch_first=True, # 是否传入是把batch放在第一个纬度
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None) # 使用rnn方法会返回两个量，一个是输出量，一个是对于之前图片的记忆
        out = self.out(r_out[:,-1,:]) # (batch, time_step, input)
        return out

rnn=RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
```
## RNN神经网络
***
```python
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate

# show data
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__ (self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward (self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out [:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None  # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np [np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np [np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # rnn output
    # !! next step is important !!
    h_state = h_state.data  # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)  # calculate loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)

plt.ioff()
plt.show()
```
## Autoencoder
***
这是一种无监督学习，不依赖于标签，通过自发学习将输出结果与原结果进行损失计算后重复训练，样本既是输入又是评判标准。
```python
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch_Variable import Variable
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=12),
            nn.Tanh(),
            nn.Linear(in_features=12, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=28*28),
            nn.Sigmoid(), # 不太明白为什么要压缩到0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()
view_data = train_data.data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step , (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))
        b_y = Variable(x.view(-1, 28*28))
        b_label =  Variable(y)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 ==0:
            print('Epoch:',epoch,'| train loss:%.4f')
            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a [1] [i].clear()
                a [1] [i].imshow(np.reshape(decoded_data.data.numpy() [i], (28, 28)), cmap='gray')
                a [1] [i].set_xticks(());
                a [1] [i].set_yticks(())
            plt.draw();
            plt.pause(0.05)

plt.ioff()
plt.show()

# visualize in 3D plot
view_data = train_data.train_data [:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2);
ax = Axes3D(fig)
X, Y, Z = encoded_data.data [:, 0].numpy(), encoded_data.data [:, 1].numpy(), encoded_data.data [:, 2].numpy()
values = train_data.train_labels [:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255 * s / 9));
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max());
ax.set_ylim(Y.min(), Y.max());
ax.set_zlim(Z.min(), Z.max())
plt.show()
```

## 写在最后
pytorch非常年轻，某些特性在torch的更新过程中已经变得更智能，或者说更贴合实际的需求，文中的代码有一些是比较繁缛的。

注：代码部分引用于[MorvanZhou](https://github.com/MorvanZhou/PyTorch-Tutorial/tree/master/tutorial-contents)
