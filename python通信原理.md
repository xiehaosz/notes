# 通信原理



实现OFDM

https://blog.csdn.net/weixin_43935696/article/details/121407563



Python信号处理：信道、波束形成及目标方位估计，CBF、MVDR

https://commpy.readthedocs.io/en/latest/index.html#



## I/Q数据的生成与解调

原理：https://zhuanlan.zhihu.com/p/102144412

代码演示

## 傅里叶级数合并方波

代码演示

## 方波三角波的合成与分析

https://blog.csdn.net/whujk/article/details/108541324

https://blog.csdn.net/weixin_42066185/article/details/103600532



## 傅里叶级数展开

https://blog.csdn.net/Night_MFC/article/details/84669194

https://stackoverflow.com/questions/32590720/create-scipy-curve-fitting-definitions-for-fourier-series-dynamically

代码演示

## 旋转向量

代码演示

## 李萨育图形

代码演示

## 正/余弦复合函数

代码演示

## SINC函数

代码演示

## 天线方向图

代码演示

## 天线方向图3D

代码演示

## 综合样例网

http://liao.cpython.org/scipytutorial18/





# 数学方法

## Python和matlab函数对应

https://blog.csdn.net/panghaomingme/article/details/70308290



## 卷积

假设B是一个**因果系统**，其t时刻的输入为**因果信号**x(t)，输出为y(t)，输出为h(t)，按理说y(t) = x(t) h(t)。

但实情况是系统的输出不仅与当前t时刻的响应相关，还受到历史时刻τ的影响，表示为 x(τ) h(t-τ)，这个过程可以是离散的或连续的，所以t时刻的输入为t时刻之前系统响应函数在各个时刻的响应的叠加，就是卷积。从数学角度来看，卷积是两个函数乘积的积分连续函数)或者求和(离散函数)。从运算过程来看，卷积是把一个函数卷(翻)过来，然后与另一个函数求内积。

https://www.zdaiot.com/MachineLearning/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E5%8D%B7%E7%A7%AF%E7%9A%84%E6%9C%AC%E8%B4%A8%E5%8F%8A%E7%89%A9%E7%90%86%E6%84%8F%E4%B9%89%EF%BC%88%E5%85%A8%E9%9D%A2%E7%90%86%E8%A7%A3%E5%8D%B7%E7%A7%AF%EF%BC%89/

代码演示

## 矩阵乘法实现

```python
def MatrixMultiply(a, b):
    n, m, p, q = len(a), len(b[0]), len(a[0]), len(b)
    if p != q:
        raise ValueError('shapes (%d,%d) and (%d,%d) not aligned' % (n, p, q, m))
    c = [[0]*m for row in range(n)] #初始化c为n行n列的全零矩阵
    for i in range(0, n):
        for j in range(0, m):
            for k in range(0, p):
                c[i][j] += a[i][k] * b[k][j]
    return c
```

## 矩阵卷积

代码演示

## 积分

```python
import numpy as np
from matplotlib import pyplot as plt

points = 1000
x = np.linspace(0, 6, points)
y = np.cos(2*np.pi*x)*np.exp(-x)+1.2
y0 = [0] * points

# 坐标范围
plt.axis([np.min(x), np.max(x), 0, np.max(y)])
# 画曲线，带图示
plt.plot(x, y, label="$cos(2πx)e^{-x}+1.2$")
plt.plot(x, y0, 'b--', linewidth=1) 

# 填充积分区域
plt.fill_between(x, y1=y, y2=0, where=(x >= 0.7) & (x <= 4), facecolor='blue', alpha=0.2)
# 增加说明文本
plt.text(0.5*(0.7+4), 0.4, r"$\int_{0.7}^4(cos(2πx)e^{-x}+1.2)\mathrm{d}x$",
         horizontalalignment='center', fontsize=14)
# 显示图示
plt.legend()
plt.show()

'''方法1：分成小矩形，计算面积和'''
# 积分区间
x = np.linspace(0.7,4.0,1000) 
#每个矩形的宽度
dx = x[1] - x[0] 
#矩形宽*高，再求和
fArea = np.sum(y*dx)                    
print("Integral area:", fArea)

'''方法2：使用quad()函数进行积分'''
import math
from scipy import integrate

def func(x):
    print("x=",x)       #用于展示quad()函数对func的多次调用
    return math.cos(2*math.pi*x)*math.exp(-x)+1.2

fArea,err = integrate.quad(func,0.7,4)
print("Integral area:",fArea)
```

python Scipy积分运算大全（integrate模块——一重、二重及三重积分）：https://www.cnblogs.com/Yanjy-OnlyOne/p/11185582.html



## 矩阵的相关性计算

代码演示

## 矩阵的SDV分解

**奇异值**往往对应着矩阵中隐含的重要信息，且重要性和奇异值大小正相关。每个矩阵A表示为一系列秩为1的“小矩阵”之和，而奇异值则衡量了这些“小矩阵”对于A的权重。https://www.zhihu.com/question/22237507

**秩**：矩阵中所有行向量中极大线性代无关组的元素个数。如果把矩阵对应到方程系数，秩就是有价值的方程个数，也就是方程能解多少个值。

矩阵的**范数** Norm 等于矩阵最大的奇异值

**特征向量**在矩阵A的作用下，方向保持不变，进行比例为lambda（**特征值**）的伸缩。

**特征向量**在A的作用下保持不变，进行比例为lambda（特征值）的伸缩。

物理意义：一个矩阵作用于矢量，可以代表某个变化，当对任意一个矢量不断反复作用时（反复运用矩阵乘法），该矢量会越来越贴合到特征空间上（即所有特征向量的集合）。

```python
# coding: UTF-8
import numpy as np
from scipy import linalg

# A=UΣVT
int_m = 4
int_n = 4

arr_a = np.mat([[2, 4], [1, 3], [0, 0], [0, 0]])
# arr_a = np.mat([[4, 18], [1, 7], [19, 4], [13, 12]])
# arr_a = np.mat([[20, 8], [9, 22], [11, 22], [18, 6]])
# arr_a = np.mat(np.random.randint(1, 25, (int_m, int_n)))
print('矩阵A: \n', arr_a)

# 方法1:使用np自带包计算U、Σ、VT
arr_u, sigma, arr_vt = np.linalg.svd(arr_a)
'''
python中svd分解得到的VT是V的转置，matlab中svd后得到的是V
Python中svd分解得到的sigma是一个行向量，Python中为了节省空间只保留了A的奇异值，需要将它还原为奇异值矩阵
'''
arr_sigma = np.zeros(arr_a.shape)
for iii in range(sigma.shape[0]):
    arr_sigma[iii][iii] = sigma[iii]

arr_a_ = arr_u * arr_sigma * arr_vt
print('自带包还原A: \n', arr_a_)


# 手动过程：
# 计算AT*A和A*AT
arr_ata = arr_a.T * arr_a
arr_aat = arr_a * arr_a.T
# 求解矩阵的特征值、特征向量
lambda1, vector1 = np.linalg.eig(arr_ata)
lambda2, vector2 = np.linalg.eig(arr_aat)

# lambda2_s, vector2_s = linalg.eig(arr_aat)
# print('numpy: \n', vector2)
# print('scipy: \n', vector2_s)

# Σ上的对角线元素由特征值的算术平方根组成，从大到小排列
if lambda1.size > lambda2.size:
    sigma_ = np.flip(np.sort(np.sqrt(lambda2)))
else:
    sigma_ = np.flip(np.sort(np.sqrt(lambda1)))

# Step3:计算U、Σ、VT
arr_u_ = vector2
arr_vt_ = vector1.T  # 对V矩阵进行转置
arr_sigma_ = np.zeros(arr_a.shape)
for iii in range(sigma_.shape[0]):
    arr_sigma_[iii][iii] = sigma_[iii]

arr_a_ = arr_u_ * arr_sigma_ * arr_vt_
print('手动计算还原A: \n', arr_a_)

# # 手动计算有时候会不一样，对比
print('自动U: \n', arr_u)
print('手动U: \n', arr_u_)
print('自动S: \n', sigma)
print('手动S: \n', sigma_)
print('自动V: \n', arr_vt.T)
print('手动V: \n', arr_vt_.T)

'''
当SVD结果的矩阵出现以下这两种情况时：
1、需要整个矩阵添加相反的正负号才等于原矩阵；
2、SVD求出的矩阵中某一列（多列也一样）需要添加相反的正负号才等于原矩阵；
以上两种情况都认为：你求解SVD成功，求的这个SVD和原矩阵一致。

解SVD常遇见的疑问
1、需要添加正负符号问题
在例题2中出现了最后的矩阵需要再绝对值才等于原矩阵，或者出现某一列需要添加正负正反的符号才等于原矩阵，
为什么会这样？
答：原因是你所用的计算软件对“它”动的毛手毛脚。
我非常肯定这一点。在查阅网上大量相似问题又最后无解，自己在matlab中发现：如果你只是一步一步求解 
AA^{T} 或者 A^{T}A ，最后将结果分别依照公式拼凑上去，那么问题就在你求解 AA^{T} 和 A^{T}A 的特征向量时，
计算软件有时会给某一列添加和你预想结果完全每一个都相反的符号。
所以在最后得出的计算结果中，将会出现某一列或者整个矩阵出现需要添加正负号才等于原矩阵的原因。
但是，这并不干扰你求解SVD的正确性。它给你添加的相反符号也没有错。
因为它们本质是“特征向量”，本质是线性或非线性方程的基础解析，只要它是相应的线性方程组的基础解析就没问题。
同一特征值的特征向量的非零线性组合，仍然是这个特征值的特征向量。比如，若 x 是方阵A某个特征值对应的特征向量，
则kx 也是A对应于该特征值的特征向量。（k取正数或者负数，不取0）在之前的章节里，
很清楚得说明白了k是基础解析的k倍，基础解析的k倍，无论k几倍都是某个特征值的特征向量，
它们都存在在一条轴上，一条由同一个方向上各个长短不同的特征向量组成的“旋转轴”。
所以对某一列向量，或者所有的列向量添加相反的符号（乘于-1），得到的SVD分解结果都是正确的。
但若非整个列向量取相反符号，而是仅仅存在某一个分量有符号问题，那就是你求解中出现了问题。
但是！这种情况却有一个例外，请参考下一点 2

2、“求错”的地方刚好撞见“转置”
比如，在麻省理工大学Gilbert Strang教授讲的《线性代数》的奇异值分解那一课，他出现错误的问题在于，
“求错”的地方刚好撞见“转置”，V矩阵正确的符号应该是：
正 负
正 正
他求成了：
正 正
正 负
本来这样也是没错的，因为在1中，我说特征向量kx和x一样，在上面两个矩阵中，下面的列2只不过乘于-1就
等于上面的列2，本来是正确的，但是问题就在于，V矩阵最后在SVD公式里需要转置，而转置的位置刚好就是将E12变到E21，
将E21变到E12，所以原本在属于那一列中没有错误的计算，转置后就错误了。解决的方法并不是手解，
我想没人蠢到用手解解一个"万阶矩阵"吧？有的话请告诉我。。。解决的方法在下一段将出现。
（下一段为：计算软件求解SVD的要注意的问题）

3、其它常见出错的问题
有可能是V没有进行转置：反正我是看到Gilbert Strang教授那一课中，其实不仅没有发现“错误点”在转置位置，而且，
他还忘了对V进行转置。（所幸当时那个矩阵转不转置两者都一样，所以他忘了也没什么。解错，这不毫不奇怪，
但可笑是，国内的相关文章上都依照他错误的方式去解，不仅没有解答真正的本质在于转置位置，
就连V需要转置这么基础的也没有一个人记得。我看过很多文章，大家都生搬硬抄直接将黑板的写上去，
完全不怀疑他有没有可能写错了。真是读死书。也怪不得国外有Wolframalpha、Mathpad、Mathematics等等软件，
国内却只能在每个软件的评论区喊着“有没有中文版？”）

有可能是U的符号出现错误
在多次实验当中，我发现如果最后的数字出现了错误，那么很可能是因为左乘矩阵出现了问题，
在放到这个SVD的例子当中，SVD的左乘矩阵就是U。

也就是可以这么总结：
如果只是列向量的正负出现问题，要么你是在V上出心大意忘了添加某个符号，要么就是你刚好碰上"错误点"和"转置位置"；
如果你发现整个SVD结果的矩阵数字都出现问题了，数字都完全不相同了，而且可能正负号都不按规律来，那么，首先检查左乘矩阵U矩阵的正负符号。

计算软件求解SVD的要注意的问题
首先，用哪一款软件计算SVD都不是重点，重点是：不要一个一个去求，即是不要去求 AA^{T} 和 AA^{T} ，
然后再一一去求各自的特征向量组成V、U。千万不要这样去求！！！
因为有可能出现上面出现常见问题1中描述的，当正确而又不恰当的特征向量的分量出现在V的E12转置位置时，
有问题出现。所以，我们在用计算软件求解SVD时，请直接使用求解SVD的命令。而不要一一求解，再自己去组合计算。
'''

```

```
def singular_percentage(arr, svd_sigma, traget):
    i = 0
    while np.sum(svd_sigma[:i])/np.sum(svd_sigma) < traget:
        i += 1
    svd_sigma[i:] = 0
    l = len(svd_sigma)
    arr = np.zeros_like(arr)
    arr[:l,:l] = np.diag(svd_sigma)
    return arr

def svd_recover(svd_u, svd_sigma, svd_vt):
    return np.round(np.dot(np.dot(svd_u,svd_sigma),svd_vt),decimals=2)


def multi_dot(arr_list):
    res = arr[0]
    for i in range(1,len(arr)):
        res = np.dot(res, arr[i])

arr = np.array([[1,0,0,0,0,1],[1,1,1,1,1,1], [0,1,1,1,1,0],[0,0,1,1,0,0],
                [0,0,1,1,0,0],[0,1,1,1,1,0],[1,1,1,1,1,1],[1,0,0,0,0,1]])
print('原始矩阵:\n', arr)

svd_u, svd_sigma, svd_vt = linalg.svd(arr)
print('singular:', svd_sigma)
target = 90  # 100%
print('%d%% 奇异值还原:' % target)
print(svd_recover(svd_u, singular_percentage(arr, svd_sigma, target/100), svd_vt))


n_arr = np.where(arr > 0, arr, np.round(np.random.rand(*arr.shape)/20,decimals=2))
print('噪声影响:\n', n_arr)
n_svd_u, n_svd_sigma, n_svd_vt = linalg.svd(arr)
target = 90  # 100%
print('%d%% 奇异值还原:' % target)
print(svd_recover(n_svd_u, singular_percentage(arr, n_svd_sigma, target/100), n_svd_vt))


rhh = np.dot(n_arr.T, n_arr)
r_svd_u, r_svd_sigma, r_svd_vt = linalg.svd(arr)
target = 90  # 100%
print('%d%% 协方差还原:' % target)
print(svd_recover(n_svd_u, singular_percentage(arr, r_svd_sigma, target/100), n_svd_vt))
# print(svd_recover(svd_u, r_svd_sigma, svd_vt))

print('H(HhH)-1')
res = np.dot(n_arr,np.linalg.inv(np.dot(n_arr.T,n_arr)))
print(res/np.max(np.abs(res)))
```





## 正弦量的向量表示

https://wenku.baidu.com/view/bc5eaa5c29ea81c758f5f61fb7360b4c2f3f2a32.html

https://wenku.baidu.com/view/570ba99fa7c30c22590102020740be1e650ecce7.html

## 复矩阵运算

代码演示

## 范数

https://stackoverflow.com/questions/26680412/getting-different-answers-with-matlab-and-python-norm-functions

```python
# coding: UTF-8
import numpy as np
from scipy import linalg

a = np.mat([[0.9940, 0.0773, -0.0773],
            [-0.0713, 0.9945, 0.0769],
            [0.0828, -0.0709, 0.9940]])

b = np.mat([0.9940, 0.0773, -0.0773])

print(np.linalg.norm(a))
print(np.linalg.norm(a, 2))

print(np.linalg.norm(b))
print(np.linalg.norm(b, 2))
```



## 傅里叶变换

https://zhuanlan.zhihu.com/p/77271148

https://blog.csdn.net/shenziheng1/article/details/52862606/

https://blog.csdn.net/qq_27825451/article/details/88553441

https://zhuanlan.zhihu.com/p/98399177

https://zhuanlan.zhihu.com/p/19759362?columnSlug=wille

https://www.ilovematlab.cn/thread-541003-1-1.html



### 离散傅里叶变换的理解

代码演示

### 快速傅里叶变换

代码演示

### 快速傅里叶逆变换

```python
import numpy as np
import matplotlib.pyplot as plt


def signal_samples(t,f0,f1):
    # 产生一个信号，由两个正弦波叠加而成，两个正弦波一个频率为1Hz另一个正弦波的频率为20Hz
    return np.sin(2 * np.pi * f0 * t) + np.sin(2 * np.pi * f1 * t)


B = 3.0
f_s = 2 * B
delta_f = 0.01
N = int(f_s / delta_f)
T = N / f_s
t = np.linspace(0, T, N)  # 采样序列

f0, f1 = 1, 20
f_t = signal_samples(t, f0, f1)
s0 = np.sin(2 * np.pi * f0 * t)
s1 = np.sin(2 * np.pi * f1 * t)

# fig, axes = plt.subplots(1, 1, figsize=(16, 3), sharey=True)
# axes.plot(t, f_t)
# axes.plot(t, s0)
# axes.plot(t, s1)
# axes.set_xlabel("time (s)")
# axes.set_ylabel("signal")
# axes.set_title('smple signal')
# plt.show()

from scipy import fftpack
F = fftpack.fft(f_t)
f = fftpack.fftfreq(N, 1.0/f_s)
F_filtered = F * (abs(f) < 10)
f_t_filtered = fftpack.ifft(F_filtered)
mask = np.where(f >= 0)

fig, axes = plt.subplots(3, 1, figsize=(8, 6))
axes[0].plot(f[mask], np.log(abs(F[mask])), label="real")
axes[0].plot(B, 0, 'r*', markersize=10)
axes[0].set_ylabel("$\log(|F|)$", fontsize=14)
axes[1].plot(f[mask], abs(F[mask])/N, label="real")

axes[1].set_ylabel("$|F|$", fontsize=14)
axes[2].plot(t, f_t, label='original')
axes[2].plot(t, f_t_filtered.real, color="red", lw = 3, label='filtered')
axes[2].set_xlim(1, 11)
axes[2].set_xlabel("time (s)", fontsize=14)
axes[2].set_ylabel("$|F|$", fontsize=14)
plt.show()
```

### 傅里叶变换源代码

https://www.jianshu.com/p/0bd1ddae41c4

https://blog.csdn.net/Dr_maker/article/details/107841986

```python
from cmath import sin, cos, pi

class FFT_pack():
    def __init__(self, _list=[], N=0):  # _list 是传入的待计算的离散序列，N是序列采样点数，对于本方法，点数必须是2^n才可以得到正确结果
        self.list = _list  # 初始化数据
        self.N = N
        self.total_m = 0  # 序列的总层数
        self._reverse_list = []  # 位倒序列表
        self.output =  []  # 计算结果存储列表
        self._W = []  # 系数因子列表
        for _ in range(len(self.list)):
            self._reverse_list.append(self.list[self._reverse_pos(_)])
        self.output = self._reverse_list.copy()
        for _ in range(self.N):
            self._W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** _)  # 提前计算W值，降低算法复杂度

    def _reverse_pos(self, num) -> int:  # 得到位倒序后的索引
        out = 0
        bits = 0
        _i = self.N
        data = num
        while (_i != 0):
            _i = _i // 2
            bits += 1
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total_m = bits - 1
        return out

    def FFT(self, _list, N, abs=True) -> list:  # 计算给定序列的傅里叶变换结果，返回一个列表，结果是没有经过归一化处理的
        """参数abs=True表示输出结果是否取得绝对值"""
        self.__init__(_list, N)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        if abs == True:
            for _ in range(len(self.output)):
                self.output[_] = self.output[_].__abs__()
        return self.output

    def FFT_normalized(self, _list, N) -> list:  # 计算给定序列的傅里叶变换结果，返回一个列表，结果经过归一化处理
        self.FFT(_list, N)
        max = 0   # 存储元素最大值
        for _ in range(len(self.output)):
            if max < self.output[_]:
                max = self.output[_]
        for _ in range(len(self.output)):
            self.output[_] /= max
        return self.output

    def IFFT(self, _list, N) -> list:  # 计算给定序列的傅里叶逆变换结果，返回一个列表
        self.__init__(_list, N)
        for _ in range(self.N):
            self._W[_] = (cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** (-_)
        for m in range(self.total_m):
            _split = self.N // 2 ** (m + 1)
            num_each = self.N // _split
            for _ in range(_split):
                for __ in range(num_each // 2):
                    temp = self.output[_ * num_each + __]
                    temp2 = self.output[_ * num_each + __ + num_each // 2] * self._W[__ * 2 ** (self.total_m - m - 1)]
                    self.output[_ * num_each + __] = (temp + temp2)
                    self.output[_ * num_each + __ + num_each // 2] = (temp - temp2)
        for _ in range(self.N):  # 根据IFFT计算公式对所有计算列表中的元素进行*1/N的操作
            self.output[_] /= self.N
            self.output[_] = self.output[_].__abs__()
        return self.output

    def DFT(self, _list, N) -> list:  # 计算给定序列的离散傅里叶变换结果，算法复杂度较大，返回一个列表，结果没有经过归一化处理
        self.__init__(_list, N)
        origin = self.list.copy()
        for i in range(self.N):
            temp = 0
            for j in range(self.N):
                temp += origin[j] * (((cos(2 * pi / self.N) - sin(2 * pi / self.N) * 1j)) ** (i * j))
            self.output[i] = temp.__abs__()
        return self.output


if __name__ == '__main__':
   list = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
   a = FFT_pack().FFT(list, 16, False)
   print(a)

'''下面对源代码的一些关键部分进行解释：
我直接将计算方法的调用封装成了一个类，这样可以很方便进行扩展和调用。在FFT_pack（）这个类之中，我定义了下面的几种方法
1.构造函数__init__()，目的是初始化需要进行FFT变换的序列，对采样点数进行赋值、计算分治算法需要进行分组的层数，计算旋转因子的系数列表等。
2._reverse_pos()方法，是为了获得位倒序后的顺序，这个方法是一个不需要外部调用的方法。
3.FFT()方法，顾名思义，是计算给定序列的快速傅里叶变换结果。里面涉及到四个参数，在实际调用的时候需要传入三个参数-list N abs。_list参数是需要进行FFT运算的列表，N参数是参加计算的点的个数，注意： 这里面N的值必须是2的m次方，例如N可以是2、4、8、16、32、……1024、2048、4096等这样的数，如果填入了别的数，无法得到正确的计算结果。 abs参数是缺省值为True的参数，当abs赋值为True或者使用缺省值的时候，返回的FFT运算结果是取绝对值以后的序列，当abs参数赋值为False时，返回的FFT运算结果就是一个复数，含有实部和虚部的值。
4.FFT_normalized()方法，用法与FFT()方法类似，只不过没有abs参数，方法的返回值是经过归一化处理的FFT变换结果。
5.IFFT()方法，返回给定序列的快速傅里叶逆变换的序列。
6.DFT()方法，返回给定序列的离散傅里叶变换序列，返回结果是经过取绝对值运算的。这个DFT（）方法主要是用来与FFT方法的运算性能进行对比的，实际使用中还是使用FFT方法。'''
```

### 时域乘法与频域的循环卷积

```python
x = np.array([1, 2, 3, 4])
y = np.array([-3, 5, -4, 0])
xy = x*y

Xf = np.fft.fft(x)
Yf = np.fft.fft(y)
N = Xf.size    # or Yf.size since they must have the same size
conv = np.convolve(Xf, np.concatenate((Yf,Yf)))
conv = conv[N:2*N]
inverse_fft_xy = np.fft.ifft(conv) / N

print(xy)

print(inverse_fft_xy)
```
