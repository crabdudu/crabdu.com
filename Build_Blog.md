---
title: "Hugo与nginx快速建站"
date: 2023-03-01T02:33:04-05:00
draft: false
tags: ["Hugo","Linux"]
categories: ["Notes"]
keywords:
- Hugo
- Linux
- Nginx
description: "利用Hugo与Nginx快速部署静态网页"
---
# 使用Hugo和Nginx建站


之前的网站一直通过Wordpress或者宝塔一类的轻量级应用管理，但对服务器的运行压力较大。于是开始自己做全栈，从零开始写html、css、js，这并不是一个好主意，学习成本很高。

结合自己需求——不需要后端数据请求。于是选择Hugo作为生成器，用Nginx作为web服务器，搭配使用Markdown语法和图床部署了纯静态网站。
## Hugo
Hugo本身只是一个网站生成器，可以通俗的理解为按照Hugo的要求摆好内容，Hugo帮你生成网站框架。
### Hugo的安装
***
[Hugo](https://gohugo.io/)的安装方式取决于服务端，既可以通过包管理器安装也可以通过Go语言编译安装，一般而言在[项目地址中](https://github.com/gohugoio/hugo/releases)挑选一个合适的版本即可。

*注意：Hugo有extended版本和standard版本。前者对于某些高级的css能够更好的实现，体现在后面选择主题时，有一些主题会对Hugo的版本提出要求。*

我选择了extended版本，然后将其解压缩后添加到环境变量路径中即可。

在Hugo官方所提供的[themes list](https://themes.gohugo.io/)中挑选喜爱的主题，并按照作者要求配置toml文件，然后将整个theme里的文件替换原来的目录。

### Hugo的基本操作
***
1. #### 新建一个新网站
`hugo new site mysite.com`
此命令会在当前层级下生成一个`mysite.com`文件夹，内部包含了静态网站的各部分内容——例如content、static、data

2. #### 新建一篇文章
`hugo new posts/mypost.md`
此命令会在content目录下生成一个posts文件，其中含有mypost.md这个文件，并默认设置为草稿。

当确认编辑好文章后，将草稿设为False，最后再使用`Hugo`生成public文件夹即可。


## DNS解析配置

在域名服务商的DNS中设置好A记录解析，并指向服务器后耐心等待DNS刷新即可。
***

## Nginx
***
Nginx直接通过yum或者apt等包管理器安装，主要需要配置/etc/nginx/nginx.conf文件，根据自己的需求配置（例如配置SSL证书、多域名跳转、反向代理。*其中root路径需指向 `/path/to/public*`

使用`nginx -t`用以检查配置文件是否正确，若未通过，可以查询nginx error日志解决。

使用`nginx -s reload`用以重新加载nginx服务使用新的配置,或者使用`systemctl restart nginx`重启nginx服务。

## Certbot
***
配置好以上文件，再设置好DNS解析后，即可使用http访问网站了，为了提高安全性通常需要配置SSL证书。

我选择使用免费的Certbot服务。

根据[certbot](https://certbot.eff.org/)官网要求，尽量通过Snapd安装Certbot，这样Certbot可以直接识别你使用的web服务并自动重写nginx.config文件。

最后使用`certbot renew --dry-run`完成SSL证书自动取得即可。



## 后记
- hugo是一个开源项目，在兼容性上或多或少受到了影响，新旧主题与新旧hugo版本总是会产生奇妙的error，甚至有可能在两方的更新中，突然某一方就出现了不可用问题。
- nginx的配置最好参照default文件慢慢更改，且最好不要一次性大幅更改。