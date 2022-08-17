数据科学模块 NumPy

统计与分析模块 Pandas

数据可视化 Matplotlib 与 Seaborn

数据分析好助手 Jupyter notebook

GUI Tkinter：第一份是Tkinter简明教程，不知所云，几乎没什么帮助；第二份是2014年度辛星Tkinter教程第二版，内容浅显易懂；第三份是Python GUI Programming Cookbook，内容详细，帮助很大。大概用了5-6天的时间，边看文档写出了带有简单GUI界面的Anki2.0。又经过之后的组件升级，增加了许多功能，更新到了Anki3.2  https://www.zhihu.com/question/32703639

## pprinter 

数据美化输出https://blog.csdn.net/u013061183/article/details/79094460

```python
'''
class pprint.PrettyPrinter
indent=1, 每一级嵌套缩进的空格数
width=80, 默认的行宽度参数为 80，当打印的字符超过80时会做美化
depth=None 打印嵌套结构的最大深度，超出的层级会用 ... 表示，例如[1, [2, [3, [...]]]]
stream=None
compact=True 如果为true，则在宽度内将多个项目合并为一行，默认为false

pprint.pformat与pp.pprint用法相同，返回格式化后的str
'''

import pprint
stuff = [['spam', 'eggs', 'lumberjack', 'knights', 'ni'],['spam', 'eggs', 'lumberjack', 'knights', 'ni']]
pp = pprint.PrettyPrinter(indent=1, width=80)
pp.pprint(stuff)

pp = pprint.PrettyPrinter(indent=3, width=160)
pp.pprint(stuff)
```



## pandas

Pandas的内存使用

https://blog.csdn.net/weiyongle1996/article/details/78498603

数据合并

https://blog.csdn.net/qq_41853758/article/details/83280104

筛选：

https://blog.csdn.net/AlanGuoo/article/details/88874742

https://blog.csdn.net/qq_38727626/article/details/100164430

pandas直接作图

https://zhuanlan.zhihu.com/p/58410775

pandans的遍历

https://zhuanlan.zhihu.com/p/80880493

https://zhuanlan.zhihu.com/p/97269320

Pandas获取行列数据

https://www.cnblogs.com/nxf-rabbit75/p/10105271.html



csv文件：https://www.cnblogs.com/traditional/p/12514914.html

https://zhuanlan.zhihu.com/p/101284491  精品链接

```python
# coding: UTF-8
import pandas as pd
import numpy as np

csv_full_name = r'D:\Python\PycharmProjects\d_pro_python\datebase\test.csv'

company = ["a","b","c"]
country = ["A","B","C"]

data=pd.DataFrame({
    "company":[company[x] for x in np.random.randint(0,len(company),10)],
    "country":[country[x] for x in np.random.randint(0,len(country),10)],
    "salary":np.random.randint(5,50,10),
    "age":np.random.randint(15,50,10)
}
)

df_group = data.groupby('company')

# # 打印分组的方法1
# for key, itgrouped_dfem in df_group:
#     print(df_group.get_group(key))
#
# # 打印分组的方法2
# for key_of_group, group in df_group:
#    print(key_of_group, group)

# 打印分组的方法3
# df_group.apply(print)

# 获取一个分组
print(df_group.get_group('c'))
```



NaN值的处理

```
"""
NaN值的处理

统计NaN：df.isnull().sum().sum()
.isnull() 方法返回一个大小和 store_items 一样的布尔型 DataFrame，并用 True 表示具有 NaN 值的元素，用 False 表示非 NaN 值的元素
第一个 sum() 返回一个 Pandas Series，其中存储了列上的逻辑值 True 的总数
第二个 sum() 将上述 Pandas Series 中的 1 相加

删除NaN：df.dropna(axis = 0)
如果 axis = 0，.dropna(axis) 方法将删除包含 NaN 值的任何行
如果 axis = 1，.dropna(axis) 方法将删除包含 NaN 值的任何列

替换NaN：store_items.fillna(0)
前向填充值（取上一个非Na值）,axis指定轴向，0表示列，1表示行：df.fillna(method = 'ffill', axis = 0)
后向填充值（取上一个非Na值）,axis指定轴向，0表示列，1表示行：df.fillna(method = 'bfill', axis = 0)
注意：.fillna() 方法不在原地地替换（填充）NaN 值。也就是说，原始 DataFrame 不会改变。你始终可以在 fillna() 函数中将关键字 inplace 设为 True，在原地替换 NaN 值。

插值替换NaN：
线性插值：df.interpolate(method = 'linear', axis = 0)

https://blog.csdn.net/Tyro_java/article/details/81396000
https://blog.csdn.net/AlanGuoo/article/details/77198503
https://blog.csdn.net/missyougoon/article/details/83443361
https://blog.csdn.net/weixin_39750084/article/details/81750185
"""
```



## Matplotlib



图形 PDF

https://blog.csdn.net/ling620/article/details/120041440



入门：https://zhuanlan.zhihu.com/p/93423829

### 画图基础



隐藏坐标轴

https://www.delftstack.com/zh/howto/matplotlib/hide-axis-borders-and-white-spaces-in-matplotlib/



动态图表

http://www.4k8k.xyz/article/u013950379/87936999

```python
import matplotlib.pyplot as plt
import numpy as np

ax=[]   #保存图1数据
ay=[]
bx=[]   #保存图2数据
by=[]
num=0   #计数
plt.ion()    # 开启一个画图的窗口进入交互模式，用于实时更新数据
# plt.rcParams['savefig.dpi'] = 200 #图片像素
# plt.rcParams['figure.dpi'] = 200 #分辨率
plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
plt.rcParams['font.sans-serif']=['SimHei']   #防止中文标签乱码，还有通过导入字体文件的方法
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.5   #设置曲线线条宽度
while num<100:
    plt.clf()    #清除刷新前的图表，防止数据量过大消耗内存
    plt.suptitle("总标题",fontsize=30)             #添加总标题，并设置文字大小
    g1=np.random.random()  #生成随机数画图
	#图表1
    ax.append(num)      #追加x坐标值
    ay.append(g1)       #追加y坐标值
    agraphic=plt.subplot(2,1,1)
    agraphic.set_title('子图表标题1')      #添加子标题
    agraphic.set_xlabel('x轴',fontsize=10)   #添加轴标签
    agraphic.set_ylabel('y轴', fontsize=20)
    plt.plot(ax,ay,'g-')                #等于agraghic.plot(ax,ay,'g-')
	#图表2
    bx.append(num)
    by.append(g1)
    bgraghic=plt.subplot(2, 1, 2)
    bgraghic.set_title('子图表标题2')
    bgraghic.plot(bx,by,'r^')

    plt.pause(0.4)     #设置暂停时间，太快图表无法正常显示
    if num == 15:
        plt.savefig('picture.png', dpi=300)  # 设置保存图片的分辨率
        #break
    num=num+1

plt.ioff()       # 关闭画图的窗口，即关闭交互模式
plt.show()       # 显示图片，防止闪退
```



https://jishuin.proginn.com/p/763bfbd346fc

简单 https://www.163.com/dy/article/FT6HSU5505318EB9.html





保存PDF

```python
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['legend.fontsize'] = 10

#
with PdfPages('test.pdf') as pdf:
    #第一页
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection = "3d")

    #
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    ax.plot(x, y, 1.0, label='parametric curve', zdir = 'x')
    ax.legend()
    plt.title('Page1')

    pdf.savefig(fig)    
    plt.close()

    #第二页
    fig = plt.figure() 

    # 3-D投影创建 
    ax = plt.axes(projection ='3d') 

    # 定义3个坐标轴的数据 
    z = np.linspace(0, 1, 100) 
    x = z * np.sin(25 * z) 
    y = z * np.cos(25 * z) 

    # 连接所有点成曲线 
    ax.plot3D(x, y, z, 'green') 
    ax.set_title('3D line plot')

    pdf.savefig(fig)
    plt.close()
```



子图 ：https://zhuanlan.zhihu.com/p/404145594

非常多例子：https://blog.csdn.net/xiaodongxiexie/article/details/53123371

https://zhuanlan.zhihu.com/p/109245779

```python
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
x = np.arange(0,10,1)   #这个函数的第三个参数表示的是步长，以此进行划分
z = x**2
y = np.linspace(1,10,10)  #这个函数的第三个参数表示的是用几个点去划分，作为y的值

plt.plot(x,z,color = 'red',linewidth=1.0,linestyle='--')
#线颜色   线宽   线样式

plt.title(u'方法一')        #设置标题
plt.xlabel('X')             #设置x，y轴的标签
plt.ylabel('Y')
plt.xlim(0,10)              #设置x,y的区间
plt.ylim(0,100)
#plt.axis([0,10,0,100])这一句可以替换以上两句
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
# 第一个参数是点的位置，第二个参数是点的文字提示。
plt.yticks([0, 20, 60, 80, 100],
          [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$readly\ good$'])
          #$表示特殊的字体，这边如果后期有需要可以上网查，空格需要转译，数学alpha可以用\来实现
ax = plt.gca()      #gca='get current axis'
# 将右边和上边的边框（脊）的颜色去掉
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax = plt.gca()
# 绑定x轴和y轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# # 定义x轴和y轴的位置
ax.spines['bottom'].set_position(('data', 10))
ax.spines['left'].set_position(('data', 2))

plt.show()
————————————————
版权声明：本文为CSDN博主「Mr_leedom」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_38725737/article/details/82667461
```

牛逼

https://matplotlib.org/2.0.2/api/pyplot_api.html

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html



#### 图表标题和说明

https://blog.csdn.net/helunqu2017/article/details/78659490

https://blog.csdn.net/u012155582/article/details/86539477

网格线：https://blog.csdn.net/weixin_41789707/article/details/81035997

颜色以及线条控制：https://www.cnblogs.com/darkknightzh/p/6117528.html

Colormap https://www.matplotlib.org.cn/gallery/color/colormap_reference.html

利用colormap让你的图表与众不同 https://blog.csdn.net/weixin_42731853/article/details/107961511

https://zhuanlan.zhihu.com/p/181615818



图表的风格库 https://www.cnblogs.com/feffery/p/15056044.html

自带主题 https://www.heywhale.com/mw/project/5f110c7494d484002d26b65c

适合科学出版的 Matplotlib 绘图主题 https://www.jianshu.com/p/391c352da151

负数柱状图

```python
# https://cloud.tencent.com/developer/article/1786604
    
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

def plot_bar_with_neg_vals(vals):
    vals = np.asarray(vals)
    min_val = np.min(vals)
    min_val += np.sign(min_val)

    def neg_tick(x, pos, min_val):
        return '%.1f' % (x + min_val if x != min_val else 0)

    # formatter = FuncFormatter(neg_tick)
    formatter = FuncFormatter(lambda x, pos: neg_tick(x, pos, min_val))
    plt.figure()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(formatter)
    plt.bar(*zip(*enumerate(-min_val + vals)))
    plt.show()

vals = [-4, -6, -8, -6, -5]
plot_bar_with_neg_vals(vals)
```



#### 子图设置

https://zhuanlan.zhihu.com/p/75276939

https://www.pythonf.cn/read/22058

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as matcoll

x = np.arange(1, 13)
y = [15, 14, 15, 18, 21, 25, 27, 26, 24, 20, 18, 16]

# 数据系列
a_lines = [[(x[i], 0), (x[i], y[i])] for i in range(len(y))]
linecoll = matcoll.LineCollection(a_lines)

fig, ax = plt.subplots()
ax.add_collection(linecoll)

plt.scatter(x, y)

plt.xticks(x)
plt.ylim(0, 30)

plt.show()

# 对于彩色点，将plt.scatter(x,y)替换为：
colours = ['Crimson', 'Blue', 'Fuchsia', 'Gold', 'Green', 'Tomato', 'Indigo', 'Turquoise', 'Brown', 'Wheat', 'Yellow',]
plt.scatter(x, y, c=colours)
```

快速换图比较：https://blog.csdn.net/robert_chen1988/article/details/80465255

```
# https://blog.csdn.net/hesongzefairy/article/details/113527780?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-0&spm=1001.2101.3001.4242
# 标记和颜色
plt.plot(x, y, marker='+', color='coral')
```



```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as matcoll

x = np.arange(1, 13)
y = np.array([15, 14, 15, 18, 21, 25, 27, 26, 24, 20, 18, 16])
# 合成数据系列格式（可选）
a_lines = [[(x[i], 0), (x[i], y[i])] for i in range(len(y))]
linecoll = matcoll.LineCollection(a_lines)

t = np.linspace(-np.pi, np.pi, 256, endpoint=True)
c, s = np.cos(t), np.sin(t)

'''
python matplotlib各种绘图类型完整总结
https://blog.csdn.net/qq_30992103/article/details/101905466
'''

# 子图数量
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# # fig, ax = plt.subplots(2,3),其中参数2和3分别代表子图的行数和列数，函数返回一个figure图像和子图ax的array列表
'''等效两个步骤：
fig = plt.figure(figsize=(10,5))  # 定义fig
ax = fig.subplots(2, 3, sharex=True, sharey=True)  # 批量子图，sharex或sharey使用相同坐标轴刻度
ax0 = fig.add_subplot(132) # 逐一添加子图，sharex或sharey使用相同坐标轴刻度
'''
# # 调整子图间距
# plt.subplots_adjust(wspace=0, hspace=0.4)

# 折线图
ax[0].set_title('x,y plot')
ax[0].plot(x, y, 'bo--')  # 颜色标记线条简写'bo--'

# 柱状图
ax[1].set_title('y,x bar')
ax[1].bar(x, y, alpha=0.7, width=0.35, facecolor='b', label='Alexnet')  # 颜色可以用RBG编码表示，如#4c72b0
# plot函数参数：x,y; linestyle,linewidth,color; marker,markersize,markerfacecolor

# 设置坐标轴上下限，完整显示图像且美观
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
dx = (x_max - x_min) * 0.1
dy = (y_max - y_min) * 0.1

ax[0].set_xlim(x_min - dx, x_max + dx)
ax[0].set_ylim(y_min - dy, y_max + dy)

# 设置坐标轴刻度
orders = ['a', 'b', 'c', 'd', 'e']
ax[1].set_xticks([1,2,3,4,5])
ax[1].set_xticklabels(orders)

# 设置轴标签
#设置横纵坐标的名称以及对应字体格式
font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
ax[1].set_xlabel('X axis', font)

ax[1].legend()

# 总标题
fig.suptitle('my chart')

# # 保存图片
# plt.savefig('demo.png', dpi=72)
plt.show()
```



```python
from pylab import *
import numpy as np
import matplotlib.pylab as plt

# 创建一个8*6点(point)的图，并设置分辨率为80
figure(figsize=(8, 6), dpi=80)

# 创建一个新的1*1的子图，接下来的图样绘制在其中的第一块中
subplot(1, 1, 1)

# 得到坐标点(x,y)坐标
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# 绘制余弦曲线，使用蓝色的、连续的、宽度为1的线条
plot(X, C, color='blue', linewidth=2.5, linestyle='-')

# 绘制正弦曲线，使用绿色的、连续的、宽度为1的线条
plot(X, S, color='green', linewidth=2.0, linestyle='-')

# 设置横轴的上下限
xlim(-4.0, 4.0)

# 设置横轴记号
xticks(np.linspace(-4, 4, 9, endpoint=True), fontproperties='Times New Roman', size=20)

# 设置纵轴记号
yticks(np.linspace(-1, 1, 5, endpoint=True))

#设置横纵坐标的名称以及对应字体格式
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

# 设置横轴标签
plt.xlabel('X axis', font)

# 设置纵轴标签
plt.ylabel('Y axis', font)

# 设置图像标题
plt.title('Demo Figure', font)

# 以分辨率72来保存图片
savefig('demo.png', dpi=72)

# 在屏幕上显示
show()
```

#### 生成棉棒图（竖线）

https://www.jianshu.com/p/86ad0b5af790

```python
import numpy as np
import matplotlib.pyplot as plt
# 生成模拟数据集
x=np.linspace(0,10,20)
y=np.random.randn(20)
# 绘制棉棒图
markerline, stemlines, baseline = plt.stem(x,y,linefmt='-',markerfmt='o',basefmt='--',label='TestStem')
# 可单独设置棉棒末端，棉棒连线以及基线的属性
plt.setp(markerline, color='k')#将棉棒末端设置为黑色

plt.legend()
plt.show()
```

例子

https://zhuanlan.zhihu.com/p/72534851



```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def plot_undirected_graph(xy, z):

    # plt.subplots()是一个函数，返回一个包含figure和axes对象的元组,其中两个分别代表子图的行数和列数。
    # 函数返回一个figure图像和子图ax的array列表
    fig, ax = plt.subplots(1, 1)

    ax.hold(True)

    # the indices of the start, stop nodes for each edge

    i, j = np.where(z)

    # an array of xy values for each line to draw, with dimensions

    # [nedges, start/stop (2), xy (2)]

    segments = np.hstack((xy[i, None, :], xy[j, None, :]))

    # the 'intensity' values for each existing edge

    z_connected = z[i, j]

    # this object will normalize the 'intensity' values into the range [0, 1]

    norm = plt.Normalize(z_connected.min(), z_connected.max())

    # LineCollection wants a sequence of RGBA tuples, one for each line

    colors = plt.cm.jet(norm(z_connected))

    # we can now create a LineCollection from the xy and color values for each

    # line

    lc = LineCollection(segments, colors=colors, linewidths=2,

    antialiased=True)

    # add the LineCollection to the axes

    ax.add_collection(lc)

    # we'll also plot some markers and numbers for the nodes

    ax.plot(xy[:, 0], xy[:, 1], 'ok', ms=10)

    for ni in range(z.shape[0]):

        ax.annotate(str(ni), xy=xy[ni, :], xytext=(5, 5),

        textcoords='offset points', fontsize='large')

        # to make a color bar, we first create a ScalarMappable, which will map the

        # intensity values to the colormap scale

        sm = plt.cm.ScalarMappable(norm, plt.cm.jet)

        sm.set_array(z_connected)

        cb = plt.colorbar(sm)

        ax.set_xlabel('X position')

        ax.set_ylabel('Y position')

        cb.set_label('Edge intensity')

    return fig, ax
```

### 3D 图形

https://zhuanlan.zhihu.com/p/147537290

轴设置

https://www.delftstack.com/zh/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/

https://blog.csdn.net/helunqu2017/article/details/78736661

https://www.cnblogs.com/why957/p/9317006.html

https://www.cnblogs.com/nxf-rabbit75/p/10965067.html



颜色

https://paul.pub/matplotlib-3d-plotting/





### 球坐标图

```python
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 40)
THETA, PHI = np.meshgrid(theta, phi)
R = np.cos(PHI**2)
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)

plt.show()
```

```python
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 1.25, 50)
p = np.linspace(0, 2*np.pi, 50)
R, P = np.meshgrid(r, p)
Z = ((R**2 - 1)**2)

# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

# Tweak the limits and add latex math labels.
ax.set_zlim(0, 1)
ax.set_xlabel(r'$\phi_\mathrm{real}$')
ax.set_ylabel(r'$\phi_\mathrm{im}$')
ax.set_zlabel(r'$V(\phi)$')

plt.show()
```



## Seaborn

Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易

https://zhuanlan.zhihu.com/p/25909753

必备神器之seaborn

https://kuiyuanzhang.github.io/2018/02/08/%E5%BF%85%E5%A4%87%E7%A5%9E%E5%99%A8%E4%B9%8Bseaborn/

Seanborn的示例数据集，需要联网

https://blog.csdn.net/weixin_41571493/article/details/82528742

```
planets = sns.load_dataset('planets')
```



Seanbor基础：https://zhuanlan.zhihu.com/p/49035741

Seaborn常见绘图总结https://blog.csdn.net/qq_40195360/article/details/86605860

Matplotlib可视化最有价值的50个图表（附完整Python源代码）：https://www.jiqizhixin.com/articles/2019-01-15-11

在一行代码中进行各种回归分析（回归结构图汇总）：https://www.pythonf.cn/read/178711



Searborn基础：https://zhuanlan.zhihu.com/p/78565158

探索性数据分析，Seaborn必会的几种图：https://cloud.tencent.com/developer/article/1665555

```python
'''seaborn.lmplot(x, y, data, hue=None, col=None, row=None, palette=None, col_wrap=None, size=5, aspect=1, markers='o', sharex=True, sharey=True, hue_order=None, col_order=None, row_order=None, legend=True, legend_out=True, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, x_jitter=None, y_jitter=None, scatter_kws=None, line_kws=None)'''

sns.set(font_scale=1.5,style="white")  # 5种主题风格,darkgrid,white,grid,dark,white,ticks

统计分析绘制图——可视化统计关系
统计分析是了解数据集中的变量如何相互关联以及这些关系如何依赖于其他变量的过程。常见方法可视化统计关系：散点图和线图。 常用的三个函数如下： - replot()

scatterplot(kind="scatter";默认)
lineplot(kind="line"，默认)

参数说明如下：

data是输入的数据集，数据类型可以是pandas.DataFrame对象、numpy.ndarray数组、映射或序列类型等。
x和y是参数data中的键或向量，指定关系图中x轴和y轴的变量。
hue也是data中的键或向量，根据hue变量对数据进行分组，并在图中使用不同颜色的元素加以区分。
size也是data中的键或向量，根据size变量控制图中点的大小或线条的粗细。
style也是data中的键或向量，根据style变量对数据进行分组，并在图中使用不同类型的元素加以区分，比如点线、虚线等。
kind指定要绘制的关系图类型，可选"scatter"(散点图)和"line"(线形图)，默认值为"scatter"。

* x,y,hue 数据集变量 变量名
* date 数据集 数据集名
* row,col 更多分类变量进行平铺显示 变量名
* col_wrap 每行的最高平铺数 整数
* estimator 在每个分类中进行矢量到标量的映射 矢量
* ci 置信区间 浮点数或None
* n_boot 计算置信区间时使用的引导迭代次数 整数
* units 采样单元的标识符，用于执行多级引导和重复测量设计 数据变量或向量数据
* order, hue_order 对应排序列表 字符串列表
* row_order, col_order 对应排序列表 字符串列表
* kind : 可选：point 默认, bar 柱形图, count 频次, box 箱体, violin 提琴, strip 散点，swarm 分散点
size 每个面的高度（英寸） 标量
aspect 纵横比 标量
orient 方向 "v"/"h"
color 颜色 matplotlib颜色
palette 调色板 seaborn颜色色板或字典
legend hue的信息面板 True/False
legend_out 是否扩展图形，并将信息框绘制在中心右边 True/False
share{x,y} 共享轴线 True/False
```



seaborn(sns) 没有你想用的颜色？自定义cmap调色盘：https://www.jianshu.com/p/2961bc740614

```
# 分类子图
sns.set(style="white",context="talk")
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)
x = np.array(list("ABCDEFGHIJ"))
y1 = np.arange(1, 11)
sns.barplot(x=x, y=y1, palette="deep", ax=ax1)  # palette :rocket husl deep  vlag
 
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Sequential")
 
y2 = y1 - 5.5
sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("Diverging")
 
sns.despine(bottom=True)
plt.setp(fig.axes, yticks=[])
plt.tight_layout(h_pad=2)
————————————————
版权声明：本文为CSDN博主「沸点数据」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_38239039/article/details/107187771
```



## graphviz

python模块graphviz使用入门：https://blog.csdn.net/LuohenYJ/article/details/106172201



## numpy

meshgrid和mgrid

https://www.cnblogs.com/shenxiaolin/p/8854197.html



### 多项式拟合

https://blog.csdn.net/tcy23456/article/details/87478096

https://blog.csdn.net/u013468614/article/details/116192778

```
z1 = np.polyfit(xxx, yyy, 7) # 用7次多项式拟合，可改变多项式阶数；
p1 = np.poly1d(z1) #得到多项式系数，按照阶数从高到低排列
print(p1)  #显示多项式
```

当加载csv文件的多列数据时可以使用unpack将加载的数据列进场解耦到不同数组中

```python
close,amount=np.loadtxt("data.csv",delimiter=",",usecols=(6,7),unpack=True)
print("收盘价：\n",close)
print("成交量：\n",amount)
```

不用科学计数法np.set_printoptions(suppress=True)

https://www.cnblogs.com/wj-1314/p/10244807.html

批量应用函数

https://blog.csdn.net/S_o_l_o_n/article/details/103032020

```
import numpy as np
```

### 随机数

```python
np.random.rand
```

### 截取位数

```
# https://www.shangmayuan.com/a/63d740bf93a4485d92925b23.html
np.round()
np.around()
np.floor()
np.ceil()
```

### 数组/矩阵

```python
# python进行矩阵计算, numpy的坑
https://my.oschina.net/u/4330611/blog/3421029
np.multiply（）、np.dot（）和星号（*）三种乘法运算的区别
https://www.cnblogs.com/xianhan/p/11197728.html

# 通过mat/array/matrix函数转换,论numpy中matrix 和 array的区别
https://blog.csdn.net/ialexanderi/article/details/73952528
    
# arr_re = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])
# arr_im = np.array([[6, 5, 4, 3, 2, 1, 0], [6, 5, 4, 3, 2, 1, 0]])
mat_re = nps.asmatrix(arr_re)

'''判断矩阵相同'''
arr_re == arr_im  # 返回每一个元素对比的数组
(mat_hc == mat_hw).all()  # 返回一个结果

# 单位阵
np.identity(3)

# 全0/1阵
np.zeros((10, 10))
np.ones((10, 10)) 

# np对象有平局mean属性 和方差属性var
arr_re.mean()
arr_re.var

#复数的相角 numpy.angle详解
https://blog.csdn.net/wzy628810/article/details/103990151
https://blog.csdn.net/qq_18343569/article/details/50416853

# numpy数组的拼接、合并
https://blog.csdn.net/qq_39516859/article/details/80666070
    
# 数组维度重塑
https://blog.csdn.net/jingyi130705008/article/details/78077250
```



## SymPy

SymPy公式：https://zhuanlan.zhihu.com/p/83822118

求积分：https://blog.csdn.net/a19990412/article/details/80574212

# VBA

```python
求出A列最后1行：Cells(Rows.Count, 1).End(3).Row
1. "Cells(Rows.Count, 1)"表示是查找A列最后一个非空单元格，按列的可以改成"Cells(1, Columns.count)"
2. "end(3)"表示的向上查找，也可以写成“end(xlup)”,还有其他3个方向，向下，向左，向右：xldown,xltoleft,xltoright
```

正则表达式 http://yshblog.com/blog/94

该对象可以通过引用Microsoft VBScript Regular Expressions 5.5。再声明定义

属性：

1）Global，是否全局匹配，若为False，匹配到一个结果之后，就不再匹配。默认False，建议设为True；

2）IgnoreCase，是否忽略大小写，默认False，建议设为False，这个会影响到正常表达式匹配；

3）Multiline，是否跨行匹配，默认False，建议设为False，这个会影响到正常表达式匹配；

4）Pattern，获取或设置正则表达式。

方法：

1）Execute，执行匹配

2）Replace，根据正确表达式全部替换

3）Test，测试正则表达式能否匹配到内容

```python
Public Function CheckNumber(str As String) As Boolean
    Dim reg As Object
    Set reg = CreateObject("VBScript.Regexp")
            
    Dim is_exist As Boolean    
    With reg
        .Global = True
        .Pattern = "\d"        
        is_exist = .Test(str)  # 判断是否匹配到数字   
    End With
    CheckNumber = is_exist
End Function
```

```python
Public Sub GetCode()
    Dim reg As Object
    Set reg = CreateObject("VBScript.Regexp")
    
    Dim str As String
    str = "编号：ABC123155 日期：2016-01-11" & _
          "编号：ABD134215 日期：2016-02-21" & _
          "编号：CBC134216 日期：2016-01-15"
    
    reg.Global = True    reg.Pattern = "[A-Z]{3}\d+"        '因为这个编号是3个大写字母和多个数字组成。可以利用代码中的表达式匹配到3个结果：ABC123155、ABD134215和CBC134216。'
    Dim matches As Object, match As Object
    Set matches = reg.Execute(str)
    
    '遍历所有匹配到的结果'
    For Each match In matches
        '测试输出到立即窗口'
        Debug.Print match
    Next
End Sub
```

```python
Public Sub GetHref()    
    Dim reg As Object    
    Set reg = CreateObject("VBScript.Regexp")        
    
    Dim str As String    
    str = "<a href='xxx1'>xxx1</a><a href='xxx2'>xxx2</a>"        
    
    reg.Global = True    
    '获取a标签中href的属性值'    
    reg.Pattern = "href='(.+?)'"        
    
    '获取匹配结果'    
    Dim matches As Object, match As Object    
    Set matches = reg.Execute(str)        
    
    '遍历所有匹配到的结果'    
    For Each match In matches        
        '测试输出子表达式到立即窗口'        
        Debug.Print match.SubMatches(0)  用元组可以一次性搞定，通过match的SubMatches集合获取元组里面的内容。轻松得到xxx1和xxx
    Next
End Sub
```

正则的特点是书写方便但是极其不便于阅读

. 匹配任意字符

/ 之后的任意特殊字符 匹配其本身 如 /. 匹配 .

0-9 以及 a-zA-Z 以及汉字字符 匹配其本身 a{5} 即a出现5次 匹配 aaaaa 大括号用法见下文
/d 匹配数字。等价于[0-9]， 25/d 匹配 250 到 259 之间的字符串

/D匹配非数字。等价于[^0-9]

/s 匹配空白，包括空格、制表符、换页符等。等价于"[/f /n /r /t /v ]"

/S 匹配非空白的字符。等价于"[^/f /n /r /t /v ]"

/w 匹配字母、数字，以及下划线。等价于"[A-Za-z0-9_]"

/W 匹配非字符数字。等价于"[^A-Za-z0-9/_]"

我们注意到，大写字母为小写字母所表示模式的补集


定义出现频率

{a,b} 其中 a b分别为相应模式出现次数的上限与下限 /d{1,4} 表示一到四位数字

{a} 其中 a 为相应模式出现次数 /d{4} 表示由任意数字组成的四位字符串

? 出现一次或不出现 Germany? 既可以匹配 German 又可以匹配 Germany

\+ 出现一次以上 a+ 既可以匹配 aaa

\* 可能不出现 出现一次 或出现多次



[] 中括号中的模式选一 如[abcd0123] 表示从abcd0123这几个字符中任意选择一个

也可以表示为 [a-d0-3]

[^] 表示 除括号中元素之外的其他所有



| 表示二选一 ma|en 可以匹配 man 也可以匹配men

() 表示 分组 Eng(lish)|(land) 可以匹配 English 也可以匹配England



^ 为字符串开头

$ 为字符串结尾

VBA图表

https://blog.csdn.net/zishendianxia/article/details/76712366

https://blog.csdn.net/zishendianxia/article/details/76358423

https://www.163.com/dy/article/EFI6565G05368KC4.html

# git

```C
//https://www.liaoxuefeng.com/wiki/896043488029600

mkdir learn_git
cd learn_git	
pwd 			// 显示当前路径
git init		// 将当前路径变为git repository, 路径下会生成.
git add	file1.txt file2.txt		// 添加文件
git commit -m "description"		// 提交仓库并进行原因说明,注意每次commit都会有一个ID:[master 1c605e2]
/* Git命令必须在Git仓库目录内执行（git init除外），在仓库目录外执行是没有意义的 */

git status				    // 查询仓库状态:是否有变动,是否提交
git diff file1.txt			// 查询某个文件的改动点
git log --pretty=oneline	// 显示从最近到最远的提交日志（版本）,--pretty=oneline参数美观
git log --pretty=oneline --abbrev-commit

git reset --hard HEAD^		// 回退版本:HEAD表示当前版本,^号表示上一个版本,^^表示上上个版本,回退的版本较旧时可以写成HEAD~10
git reset --hard 1094a		// 回退到指定的版本"1094a"是通过git log获取到每次提交的ID,只需要写前几位即可
git reflog				   // 查看命令历史，以便确定要回到未来的哪个版本
    
git checkout -- file1.txt	// 撤销某个文件的修改
    

git remote add origin git@github.com:账号名/learngit.git	// github创建learngit repo后关联本地库
git push -u origin master								// 本地库的所有内容推送到远程库,-u参数会把本地的master分支和远程的master分支关联
git push origin master									// 已关联的推送	

git remote -v		    //查看远程库信息
git remote rm <name>	//解除远程库的绑定信息

git clone git@github.com:账号名/gitskills.git	//从远程库克隆
    
git switch -c dev				//创建并切换到dev分支,等效于:git branch dev + git switch dev. 等效命令:git checkout -b dev	 
git branch <name>				//创建或者查看分支,当前分支会有*号
git switch master				//在分支间切换, 等效命令:git checkout master
git merge -m "description" dev	 //将dev分支的修改合并到当前分支
git branch -d dev				//删除分支(合并后不需要该分支了)
git branch -D <name>			//强行删除分支(未合并)

//主干修改和分支修改出现冲突,合并时会有提示,git status可以查看冲突的文件, 解决冲突就是把Git合并失败的文件手动编辑为我们希望的内容再提交
git merge --no-ff -m "merge with no-ff" dev  //Git通常使用Fast forward模式,删除分支后会丢掉分支信息,--no-ff参数保留分支信息,能看出合并历史

/*master------分支开发-------------------------------合并merge------------->
		   		   |branch-dev----------分支修改完成|                    */

    
/*master--存在bug----分支开发--------------合并bug修复---------------------合并merge>
		   		   |branch-dev------stash-------------分支修改完成|
    	  					|发现bug新增分支------修复|				 		  */
git switch -c dev  		 	 			 //在分支上开发
git stash 			    				//储存分支,此时git status是干净的,可以多次stash
git stash list							//查询储存的分支
git switch master		 				//切换到主分支
git switch -c issue-101				   	 //创建issue分支进行修复工作
git add readme.txt 
git commit -m "fix bug 101"	 			 //完成修复,假设这次commit的ID为[4c805e2]
git switch master						//切换到主分支
git merge --no-ff -m "fix 101" issue-101  //合并
git switch dev							//继续分支开发
git stash pop							//恢复存储的状态,pop等效于:git stash apply + git stash drop
git stash apply stash@{0}				 //可以恢复指定的stash
//注意:由于dev分支源自较早的master,因此同样存在bug,要在dev上修复bug
git cherry-pick 4c805e2					 //复制一个特定的提交到当前分支,即把修复bug的提交合入dev,不用把整个分支merge过来
    
git rebase								//管理功能:本地未push的分叉提交历史整理成直线
    
git tag v1.0 <commit id>						//管理功能:默认在最新的commit打标签v1.0,也可以指定commit id打标签,不带参数则查看标签
git tag -a v0.1 -m "version 0.1 released" 1094adb //带说明文字的标签
git show v1.0								   //查看标签说明
```



在函数中打印调用自己的函数

```python
s = traceback.extract_stack()
print('%s Invoked me!' % s[-2][2])
```



# 其他

## AI Learning

https://github.com/apachecn/AiLearning



[破解xlsm文件的VBA项目密码和xlsx的工作簿保护密码](https://www.cnblogs.com/zeroes/p/reset_vbaproject_password.html)

1.修改.xlsm后缀为.zip

2.使用压缩软件打开，进入xl目录找到vbaProject.bin文件，解压出来

3、使用Hex软件打开vbaProject.bin文件，查找DPB替换成DPx，保存文件，重新压缩，修改后缀名.zip为.xlsm

4、使用excel打开.xlsm文件，开发工具->查看代码，弹出错误提示框，点击“是”或“确定”。在VBAProject上点击右键，打开工程属性，重新填入密码，如：123456，点击确定。



三角函数积分方法：https://zhuanlan.zhihu.com/p/70136575



```python
# 待研
row = 12
col = 15
grid = [['.' for i in range(col)] for j in range(row)]

for aRow in range(row):
    listRow = list()

for aColumn in range(col):
    listRow.insert(aColumn, ".")
    grid.insert(aRow, listRow)
    print(*grid[aRow])
```





英语学习：

https://www.zhihu.com/question/19730085
