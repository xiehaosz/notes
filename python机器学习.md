# 机器学习

https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC2%E6%AD%A5%20-%20%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92/04.2-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95.html

## 聚类算法示例

https://zhuanlan.zhihu.com/p/126661239

```python
from sklearn import datasets
from sklearn import cluster as cl
import matplotlib.pyplot as plt
import numpy as np


def demo_cluster(option):
    # python实现聚类算法: https://blog.csdn.net/xc_zhou/article/details/88316299
    # X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    #                                     n_clusters_per_class=1, random_state=4)
    X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=3)
    # for class_value in range(2):
    #     row_ix = np.where(y == class_value)
    #     plt.scatter(X[row_ix, 0], X[row_ix, 1], s=10)
    # plt.show()

    if option == 1:
        # 亲和力传播聚类
        model = cl.AffinityPropagation(damping=0.9)
        model.fit(X)
        yhat = model.predict(X)
    if option == 2:
        # 聚合聚类涉及合并示例，直到达到所需的群集数量为止
        model = cl.AgglomerativeClustering(n_clusters=2)
        yhat = model.fit_predict(X)
    if option == 3:
        # BIRCH 聚类（ BIRCH 是平衡迭代减少的缩写，聚类使用层次结构)包括构造一个树状结构，从中提取聚类质心
        model = cl.AgglomerativeClustering(n_clusters=2)
        yhat = model.fit_predict(X)
    if option == 4:
        # DBSCAN 聚类（其中 DBSCAN 是基于密度的空间聚类的噪声应用程序）涉及在域中寻找高密度区域，并将其周围的特征空间区域扩展为群集
        # 标签为-1的类是离散点
        model = cl.DBSCAN(eps=3, min_samples=10)
        yhat = model.fit_predict(X)
    if option == 5:
        # K-均值聚类可以是最常见的聚类算法，并涉及向群集分配示例，以尽量减少每个群集内的方差
        model = cl.KMeans(n_clusters=2)
        model.fit(X)
        yhat = model.fit_predict(X)
    if option == 6:
        # Mini-Batch K-均值是 K-均值的修改版本，它使用小批量的样本对群集质心进行更新使大数据集的更新速度更快
        model = cl.MiniBatchKMeans(n_clusters=2)
        model.fit(X)
        yhat = model.fit_predict(X)
    if option == 7:
        # 均值漂移聚类涉及到根据特征空间中的实例密度来寻找和调整质心
        # Mean-shift 算法的核心思想就是不断的寻找新的圆心坐标，直到密度最大的区域
        model = cl.MeanShift(bandwidth=2)
        yhat = model.fit_predict(X)
    if option == 8:
        # OPTICS 聚类（ OPTICS 短于订购点数以标识聚类结构）是上述 DBSCAN 的修改版本。
        model = cl.OPTICS(eps=0.8, min_samples=10)
        yhat = model.fit_predict(X)
    if option == 9:
        # 光谱聚类是一类通用的聚类方法，取自线性线性代数
        model = cl.SpectralClustering(n_clusters=2)
        yhat = model.fit_predict(X)
    if option == 10:
        # 高斯混合模型总结了一个多变量概率密度函数，顾名思义就是混合了高斯概率分布
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=3)
        model.fit(X)
        yhat = model.fit_predict(X)

    clusters = np.unique(yhat)
    print(clusters)
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=12)
    plt.show()


if __name__ == '__main__':
    demo_cluster(7)
```



聚类的数据标准化

https://long97.blog.csdn.net/article/details/90549391?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_paycolumn_v3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_paycolumn_v3&utm_relevant_index=1



时间序列平滑

https://zhuanlan.zhihu.com/p/39453139



亲和

https://www.cnblogs.com/lc1217/p/6908031.html



核函数

https://blog.csdn.net/dengheCSDN/article/details/78109253



谱聚类， 图论

https://www.cnblogs.com/pinard/p/6221564.html



导数和梯度

https://www.cnblogs.com/shine-lee/p/11715033.html

机器学习与代码推导

https://github.com/luwill/Machine_Learning_Code_Implementation



异常数据识别算法

https://www.zhihu.com/question/280696035

https://github.com/yzhao062/anomaly-detection-resources#31-multivariate-data

https://zhuanlan.zhihu.com/p/58313521



https://blog.csdn.net/ygfrancois/article/details/89373430

https://zhuanlan.zhihu.com/p/142320349

深度学习的回归指标

https://aijishu.com/a/1060000000079690



数据的标准化与缩放

https://www.cnblogs.com/chaosimple/p/4153167.html



周期挖掘

https://patents.google.com/patent/CN104750830A/zh

https://blog.csdn.net/qq_17753903/article/details/85395502



# 机器学习理论（六）多项式回归

https://zhuanlan.zhihu.com/p/77555547





滑动平均

https://www.delftstack.com/zh/howto/python/moving-average-python/



最小二乘

```python
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import leastsq


def func(kd, p0, l0):
    return 0.5*(-1-((p0+l0)/kd) + np.sqrt(4*(l0/kd)+(((l0-p0)/kd)-1)**2))


# 残差的平方和是我们试图最小化的kd的函数：
def residuals(kd, p0, l0, PLP):
    return PLP - func(kd, p0, l0)


# 这里生成一些随机数据.应用时应该加载真实数据.
N = 1000
kd_guess = 3.5
p0 = np.linspace(0, 10, N)
l0 = np.linspace(0, 10, N)
PLP = func(kd_guess, p0, l0)+(np.random.random(N)-0.5)*0.1  # 真实数据
kd, cov, infodict, mesg, ier = leastsq(residuals, kd_guess, args=(p0, l0, PLP), full_output=True)

# 这是optimize.leastsq找到的最适合的kd值.
print(kd)

# 这里我们使用刚刚找到的kd的值生成PLP的值：
PLP_fit = func(kd, p0, l0)

# 以下是PLP与p0的关系图.蓝线是从数据,红线是最佳拟合曲线.
plt.plot(p0, PLP, '-b', p0, PLP_fit, '-r')
plt.show()
```



数据边界拟合

```python
# 数据和模型
def model(x, a, b, c):
    return a / np.sqrt(x + b) + c


x_res = 1000
x_data = np.linspace(0, 2000, x_res)

# true parameters and a function that takes them
true_pars = [80, 70, -5]
y_truth = model(x_data, *true_pars)
mu_prim, mu_sec = [1750, 0], [450, 1.5]
cov_prim = [[300**2, 0], [0, 0.2**2]]

# covariance matrix of the second dist is trickier
cov_sec = [[200**2, -1], [-1, 1.0**2]]

prim = np.random.multivariate_normal(mu_prim, cov_prim, x_res*10).T
sec = np.random.multivariate_normal(mu_sec, cov_sec, x_res*1).T
uni = np.vstack([x_data, np.random.rand(x_res) * 7])

# censoring points that will end up below the curve
prim = prim[np.vstack([[prim[1] > 0], [prim[1] > 0]])].reshape(2, -1)
sec = sec[np.vstack([[sec[1] > 0], [sec[1] > 0]])].reshape(2, -1)

# rescaling to data
for dset in [uni, sec, prim]:
    dset[1] += model(dset[0], *true_pars)

# this code block generates the figure above:
plt.figure()
plt.plot(prim[0], prim[1], '.', alpha=0.1, label='2D Gaussian #1')
plt.plot(sec[0], sec[1], '.', alpha=0.5, label='2D Gaussian #2')
plt.plot(uni[0], uni[1], '.', alpha=0.5, label='Uniform')
plt.plot(x_data, y_truth, 'k:', lw=3, zorder=1.0, label='True edge')
plt.xlim(0, 2000)
plt.ylim(-8, 6)
plt.legend(loc='lower left')
plt.show()

# mashing it all together
dset = np.concatenate([prim, sec, uni], axis=1)

"""
拟合点分布的边缘.常用的回归方法,如非线性最小二乘scipy.optimize.curve_fit,取数据值y并优化模型的自由参数,
使y和模型(x)之间的残差最小.非线性最小二乘是一个迭代过程,试图在每一步摆动曲线参数,以改善每一步的拟合.
现在显然,这是我们不想做的一件事,因为我们希望我们的最小化程序能够让我们尽可能远离最合适的曲线(但不要太远).
因此,让我们考虑以下功能.它不是简单地返回残差,而是在迭代的每一步也“翻转”曲线上方的点,并将它们考虑在内.
这样,曲线下面的点总是比它上面的点多,导致曲线每次迭代都向下移动！达到最低点后,找到函数的最小值,散点的边缘也是如此."""


def get_flipped(y_data, y_model):
    flipped = y_model - y_data
    flipped[flipped > 0] = 0
    return flipped


def flipped_resid(pars, x, y):
    """
    For every iteration, everything above the currently proposed
    curve is going to be mirrored down, so that the next iterations
    is going to progressively shift downwards.
    """
    y_model = model(x, *pars)
    flipped = get_flipped(y, y_model)
    resid = np.square(y + flipped - y_model)
    # print pars, resid.sum() # uncomment to check the iteration parameters
    return np.nan_to_num(resid)


# plotting the mock data
plt.plot(dset[0], dset[1], '.', alpha=0.2, label = 'Test data')

# mask bad data (we accidentaly generated some NaN values)
gmask = np.isfinite(dset[1])
dset = dset[np.vstack([gmask, gmask])].reshape((2, -1))

guesses = [100, 100, 0]
fit_pars, flag = leastsq(func=flipped_resid, x0=guesses, args=(dset[0], dset[1]))

# plot the fit:
y_fit = model(x_data, *fit_pars)
y_guess = model(x_data, *guesses)
plt.plot(x_data, y_fit, 'r-', zorder=0.9, label='Edge')
plt.plot(x_data, y_guess, 'g-', zorder=0.9, label='Guess')
plt.legend(loc='lower left')
plt.show()
```

确定多项式系数

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Perceptron
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

X = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
y = np.array(2*(X**4) + X**2 + 9*X + 2)
#y = np.array([300,500,0,-10,0,20,200,300,1000,800,4000,5000,10000,9000,22000]).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
rmses = []
degrees = np.arange(1, 10)
min_rmse, min_deg,score = 1e10, 0 ,0

for deg in degrees:
    # 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)

    # 多项式拟合
    poly_reg = LinearRegression()
    poly_reg.fit(x_train_poly, y_train)
    #print(poly_reg.coef_,poly_reg.intercept_) #系数及常数
    
    # 测试集比较
    x_test_poly = poly.fit_transform(x_test)
    y_test_pred = poly_reg.predict(x_test_poly)
    
    #mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
    poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmses.append(poly_rmse)
    # r2 范围[0，1]，R2越接近1拟合越好。
    r2score = r2_score(y_test, y_test_pred)
    
    # degree交叉验证
    if min_rmse > poly_rmse:
        min_rmse = poly_rmse
        min_deg = deg
        score = r2score
    print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse,r2score))
        
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')
ax.set_title('Best degree = %s, RMSE = %.2f, r2_score = %.2f' %(min_deg, min_rmse,score))  
plt.show()
```





Birch聚类:

1. Birch（Balanced Iterative Reducing and Clustering using Hierarchies）是层次聚类的典型代表，天生就是为处理超大规模数据集而设计的，它利用一个树结构来快速聚类，这个树结构类似于平衡B+树，一般将它称之为聚类特征树(Clustering Feature Tree，简称CF Tree)。这颗树的每一个节点是由若干个聚类特征(Clustering Feature，简称CF)组成。

https://www.cnblogs.com/pinard/p/6179132.html

