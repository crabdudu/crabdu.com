---
title: "Pytorch奥地利单站降水预报实例"
date: 2022-03-08T02:48:29-05:00
draft: false
tags: ["Meteorology","Python","DeepLearning","Pytorch"]
categories: ["Notes"]
keywords:
- Python
- DeepLearning
description: "使用深度学习预测奥地利单站降水"
---
# 数据集
***
现在我们有奥地利一些站点从1948-2008年的最高温度数据，可以用它做很多的事情，时序性预测是其中的一个选择。  

我选择将1948-2000的数据作为训练集，2000-2004的数据上使用各种方法进行改进作为验证集，最后在2004-2008上进行测试作为测试集。
## 整体思路
***
由于闰年的存在，采用了将每4年的数据作为一个样本，每次输入N——N+4年的数据，而把N+4——N+8的数据作为标签监督学习N——N+4的数据，另外模型采用了Linear层，这些并不是最好的选择，原因有两点：  
1. 以4年封装为一个样本虽然解决了输入特征数量不同的问题（若为1年则有时输入为365，而有时输入为366），但这样做的代价就是样本数变为了原来的1/4。样本数量的减少会导致精度大大下降。
2. 采用Linear作为神经层，模型并不能学习到这种**时序性**的特征，它只是将每一个输入都当做独立的个体，而事实上每个数据之间都有很强的关联。LSTM模型或许是一个更好的选择。

*不过根据效果来看，Linear模型仍然发现了时序的特点*

这并不是一个解决的很漂亮的问题，只是通过尝试解决这个简单的问题可以了解到深度学习中的基本步骤和大体思想。


```python
%matplotlib inline # 能够在jupyter中直接显示出图片
import sys #这个只是为了使用sys.exit()
import torch
import torch.nn as nn # 构造模型类的时候需要继承的模块
import torch.utils.data as Data # 为了使用dataloader进行批训练
import torch.optim as optim # 选择一个合适的参数更新器

import numpy as np
import pandas as pd

from datetime import datetime # 关于时间的处理
```
这是一堆模块的引入。  
接下来考虑数据的引进和处理，**No data, No model.**
```python
data_dir = './data/DATEN/Tmax/'
tmax_filename = 'tmax_104000_hom.txt'

tmax_df = pd.read_csv(data_dir + tmax_filename, sep='\s+', header= None, skiprows=3) # sep是数据之间的分隔符，\s+代表很多空格，header=None代表没有表头，skiprows=3代表我并不需要前三行。
```
将路径和文件名分开获取是一个好习惯，读取数据推荐使用pandas把读入的数据作为dataframe类。如果直接使用np按行读取，只会增加逻辑上的负担。  
让我们来看看读入了什么东西  
`print(tmax_df,tmax_df.shape)`  
```              0     1
0      19480101  -1.0
1      19480102  -2.0
2      19480103  11.0
3      19480104  10.0
4      19480105   5.0
...         ...   ...
22641  20091227   1.4
22642  20091228   4.8
22643  20091229  -0.3
22644  20091230   1.6
22645  20091231   3.8

[22646 rows x 2 columns] (22646, 2)
```
这是一个按时间序列排序的最高温度资料，其第一维（第[0]维）是时间，第二维（第[1]维）是最高温度。  

```python
# clean data

tmax_df.columns   = ['date', 'maxtem'] #设定列名
for i in range(len(tmax_df['date'])):
    s = str(tmax_df['date'][i]) #把int转换成str
    tmax_df['date'][i] = datetime.strptime(s, '%Y%m%d' ) # 利用strptime函数把str按固定格式转换成datetime
    tmax_df['date'][i] = tmax_df['date'][i].date() # 只关心年月日，所以去掉了time，只要date
pd.to_datetime(tmax_df['date'])
tmax_df.set_index('date')
# tmax_df
```
```
date	maxtem
1948-01-01	-1.0
1948-01-02	-2.0
1948-01-03	11.0
1948-01-04	10.0
1948-01-05	5.0
...	...
2009-12-27	1.4
2009-12-28	4.8
2009-12-29	-0.3
2009-12-30	1.6
2009-12-31	3.8
22646 rows × 1 columns
```
将时间一列从int转换为datetime类型以方便于后面画图和索引。

```python
device = (torch.device('cuda') if torch.cuda.is_available()
         else torch.device('cpu'))
print(f"Training on device {device}")
```
这是为了检查是否能够使用GPU训练

```python
## plot
# plt.style.use('fivethirtyeight')
from matplotlib import pyplot as plt
fig , ((ax1,ax2,ax3)) = plt.subplots(3,1,figsize = (15,15))
TRAIN_SIZE = 19000 # 猜了一个值，并不准确
fig.autofmt_xdate(rotation = 45)

ax1.plot(tmax_df['date'], tmax_df['maxtem'])
ax1.set_xlabel(''); ax1.set_ylabel('Max temperature'); ax1.set_title('1948-2008 Max Temperature')

ax2.plot(tmax_df['date'][0:TRAIN_SIZE], tmax_df['maxtem'][0:TRAIN_SIZE])
ax2.set_xlabel(''); ax2.set_ylabel('Max temperature'); ax2.set_title('1948-2000 Max Temperature')

ax3.plot(tmax_df['date'][TRAIN_SIZE:], tmax_df['maxtem'][TRAIN_SIZE:])
ax3.set_xlabel(''); ax3.set_ylabel('Max temperature'); ax3.set_title('2000-2008 Max Temperature')

# ax4.plot(df['date'][16329:], df['pre'][16329:])
# ax4.plot(df['date'], df['pre'])
# ax4.set_xlabel(''); ax4.set_ylabel('prediction'); ax4.set_title('2000-2008 Max Temperature Prediction')
# plt.tight_layout(pad=2)
plt.savefig('train_val_all.png')
```
可视化读入的数据  


```python
# 以4年为一样本从源数据序列时间中提取出来堆叠为15层
frontid = 0 
endid = 0 
stride = 1461
tmax_array = np.zeros([15,1461], dtype=np.float32)
for year in range(15):
    endid  = frontid + stride
    tmax_array[year] = tmax_df['maxtem'][frontid : endid]
    frontid = frontid +stride

```
其实可以使用stack函数。
```python
# 索引为0,1,2的数据即前12年的数据缺失太多，直接去掉。
# 索引为12的数据中间缺失太多，直接去掉
# 因此训练集的输入集为3-10,验证集为4-11
# 验证集的输入集为13,验证集为14
# 补充：缺失太多只是编号为10000的站点，其他站点无异常。
input_array_train = np.zeros([8,1461])
target_array_train = np.zeros([8,1461])
input_array_val = np.zeros([1,1461])
target_array_val = np.zeros([1,1461])


input_array_train = tmax_array[3:11]
target_array_train = tmax_array[4:12]
input_array_val = tmax_array[13]
target_array_val = tmax_array[14]
```
到此，在np和dataframe上的数据处理完成，接下来都是在tensor上处理
```python
# convert to tensor
input_t_train = torch.from_numpy(input_array_train)
target_t_train = torch.from_numpy(target_array_train)

input_t_val = torch.from_numpy(input_array_val).unsqueeze(0)
target_t_val = torch.from_numpy(target_array_val).unsqueeze(0)
```
***可以看见将一个形状为（8,1461）的input_array_train和target_array_train数组转换为了tensor。这很好理解，可是为什么在将input_array_val和target_array_val转换为tensor的时候需要在第0维新增一个维度？因为input_array_val和target_array_val的形状是(1461)，而神经网络是不接受一维的输入的，所有的输入必须遵循（N,Nin）的形状，无论Nin是多少维度，例如：把图片作为训练集，那么输入就是(N张图片,3,32,32)，（3,32,32）就是单张图片的形状。（N,Nin）作为输入的形状这非常重要。  
所以input_array_val和target_array_val的形状是(1461)要转成（1,1461）的形状。***
***
接下来是构造神经网络和具体的训练
```python
import torch.nn.functional as F

class Linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1461, 2048) # 输入1461个特征，输出一个2048的隐藏层
        self.linear2 = nn.Linear(2048, 512) # 输入为2048个特征，输出一个512的隐藏层
        self.linear3 = nn.Linear(512, 1461) # 输入为512个特征，输出为1461个特征
    
    def forward(self, x):
        out = self.linear1(x) #调用第一层隐藏层
        out = F.relu(out) # 在第一层的输出上使用relu激活函数
        out = self.linear2(out) #调用第二层隐藏层
        out = F.relu(out) # 在第二层的输出上使用relu激活函数
        out = self.linear3(out) # 调用第三层隐藏层
        return out
```
这个import是为了引入激活函数，激活函数是神经网络中的点睛之笔，能够让线性的东西变为非线性的东西。  
激活函数有很多的选择，要依据不同的目标来选择，选择的依据要根据不同的激活函数特性，有些时候需要一点运气。
```python
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,save_model):
    for epoch in range(n_epochs):
        for inputs,labels in train_loader: 
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(inputs.view(-1,1461))
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1000 == 0: 
            print('Epoch: %d, Loss: %f' % (epoch,float(loss)))
    if save_model == True:
           torch.save(linear_model, 'net_tmax.pkl')
           print('successfully save')
    else:
           print('unsuccessfully save')
          
```
train_loader是一个数据器，他可以将整个训练集拆分为很多的小批次，而且打乱顺序训练，一般情况下能获得更好的训练效果。而inputs从train_loader那里就拿到了第一个维度的数据，labels就从train_loader那里拿到了第二个维度的数据，具体是什么数据需要看datasets，dataloader的数据来源是自己构造的datasets  

在这里定义了一个训练循环需要怎么做：
1. 进入第一个训练周期，输入第一批训练数据，总共的批数取决于dataloader每次取多少，取得多批次就少，每次取得少训练批次就多。
2. 为了在GPU上训练，先将从loader那里拿到的数据转到GPU上去。
3. 调用模型得到输出。
4. 由损失函数计算损失。
5. 由于tensor有一个叫require_grad的参量，会选择是否记录下grad_fn，以及.grad属性，所以tensor知道自己是怎么来的，所以先将之前的梯度清零，然后通过自己的链式反向传播（BP）这些反向传播的数据都被记在了optimizer更新器里，optimizer也是自己选，不同的optimizer更新方式不一样，然后调用optimizer.step()让更新器向前更新以此更新网络参数。  

```python
# call training loop

train_dataset = Data.TensorDataset(input_t_train,target_t_train)
train_loader = Data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True)
linear_model = Linear_model().to(device=device)


n_epochs = 5000

learning_rate = 1e-3
optimizer = optim.SGD(linear_model.parameters(), lr = learning_rate)

loss_fn = nn.MSELoss()

training_loop(  
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = loss_fn,
    train_loader = train_loader,
    save_model = True
)
```
这里具体定义了dataset,将输入作为第一维，目标作为第二维构造成dataset，然后把dataset交给dataloader。  
为了能在GPU上训练，不仅数据需要在GPU，模型也需要在GPU上，所以需要将模型也移动到GPU  
调用训练循环
```python
# 验证集验证
# linear_model = torch.load("net_tmax.pkl",map_location='cuda')

val_dataset = Data.TensorDataset(input_t_val,target_t_val)
val_loader = Data.DataLoader(val_dataset, batch_size=64,
                                           shuffle=True)

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs.to(device=device)
        labels.to(device=device)
        outputs = linear_model(inputs)
        loss_val = loss_fn(outputs, target_t_val)
        print('prediction : ', outputs, outputs.shape)
        print('answer     : ', target_t_val, target_t_val.shape)
        print('loss_train: %f' % (loss_val))
```
