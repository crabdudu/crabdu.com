---
title: "风云四号卫星数据编制索引文档"
date: 2022-01-03T02:56:04-05:00
draft: false
tags: ["python", "FY-4A",]
categories: ["小玩意儿"]
description: "为FY-4A的LMI资料编制文件名-起止时间文本" 
---
## 风云四号数据介绍

风云四号搭载了LMI闪电成像探测仪，亚太地区进行持续性的闪电凝视监测。

能够以分钟级速度实时输出`.nc`文件，而这些文件通常由订单号作为索引被下载，在实际使用过程中并不方便。

通过简单的字符提取以重命名，更好的找到所需要的文件。



### 数据订单提交

手动

### 数据下载
朝曦dawn编写。下载链接[GitHub](https://github.com/zxdawn/weather_data/tree/master/FY)

也可以通过FTP连接服务器下载。


## 索引编制 订单号+开始日期————结束日期 
```python
    '''
    Write by Crabdudu Jan 2022
    请确保目录下只有需要编制索引的nc文件
    '''
import glob

def rename (filename):
    '''
    :param filename: The document that you want to find data start time and end time.(str)
    :return: starttime, endtime
    '''

    starttime = glob.glob(filename + r'\*001\*')
    starttime = starttime [0]
    starttime = starttime [98:112]
    starttime

    endtimefile = glob.glob(filename + r'\*')
    endtimefile = endtimefile [-1]
    endtime = glob.glob(endtimefile + r'\*')
    endtime = endtime [-1]
    endtime = endtime [113:127]
    endtime

    return starttime, endtime

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    path = r'nc files root path'
    filenamelist = glob.glob(path + '\*')

    for i in range(len(filenamelist)):
        if (len(filenamelist[i]) == len(path)+20) :
            # print(filenamelist[i])
            outfile = open(r'D:\\FYData\readmeplz.txt', 'a')
            starttime, endtime = rename(filenamelist [i])
            filename = filenamelist [i]
            outfile.write(filename [10:] + ':' + '   ' + starttime + '————' + endtime)
            outfile.write("\n")
            outfile.close()

```
***

