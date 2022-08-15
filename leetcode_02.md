## 黑白配**

**题目描述**

设B = {b1, b2, …, bn} 和 W = {w1, w2, …, wn}分别为平面上黑点和白点的两个集合。一黑点bi = (xi, yi) 与一白点wj = (xj, yj) 匹配当且仅当 xi <= xj 和 yi <= yj 同时成立，每个点最多只能用于一次匹配，请找出黑白点之间的最大匹配数目。
黑点和白点各自的数量均不超过100000；
平面为(0, 0)到(10000, 10000)的矩形中的整数点
黑点白点坐标可能相同，B集合、W集合中也可能包含相同的元素。

**解答要求**时间限制：2000ms, 内存限制：100MB

**输入**

输入的第一行包括一个整数T (1 <= T <= 10)，表示有T组测试数据.
对于每组测试数据：
第一行两个整数，n(黑点个数) m(白点个数)，0 < n, m <= 100000
接下来n行每行有两个整数并用空格隔开：
黑点的横坐标x 黑点的纵坐标y
再接下来m行每行有两个整数同样用空格隔开：
白点的横坐标x 白点的纵坐标y

**输出**

对于每组测试数据，输出一行一个整数，表示最大匹配数。

**样例**

输入样例 1 复制

```
2
2 2
1 0
0 1
1 1
0 0
1 1
1 1
0 0
```

输出样例 1

```
1
0
```

提示样例 1

**提示**

将所有黑白色棋子按照x轴从小到大排序，这样我们减少考虑一维。每次遇到一个黑点，将其y坐标插入平衡树中，每次遇到一个白点，寻找平衡树中小于等于该白点y坐标的黑点去掉，答案加一。因为这样去对后面是最优的。如果你去小了，可能后面比较小的白点就没有办法匹配了。

本题也是线段树的一个典型用法：对于一个动态变化的队列，并给定值x，在队列中查找不大于x的最大值。如果队列中所有的元素范围是[0,m]的话，那么使用线段树查找和更新的时间复杂度是O(logm)，线段树的空间复杂度是O(m)。如果m过大，则可以将元素值离散化，将元素的范围映射到[1,n]（n为总元素个数）。本题的m=10000,所以不需要进行离散化。

【增删元素】

在本题中，线段树的区间对应白点的纵坐标，如果要将纵坐标等于y的白点存入线段树，那么将位置y的cnt加1；反之，如果要将纵坐标等于y的白点删除，那么将位置y的cnt减1。

【查找最大值】

​    给定值y_b，需要查找线段树中不大于y_b的最大的y，那么也可以理解成查找区间[0,y_b]中的最大的y，这样就和一般线段树的查找比较相似了。但是，查找最大值的策略还是有区别的:

对于当前节点的区间[y1, y2]， 有以下3中情况：

\1.   如果其与[0,y_b]没有交集（其实就是y_b<y1），那么直接返回查找失败。

\2.   如果y1=y2，也就是叶子节点，那么查找成功，返回y1。

\3.   以上两个条件都不满足，那么先查找当前节点的右孩子，如果右孩子查找成功了，那么直接返回右孩子返回值；如果右孩子查找失败，那么再查找左孩子，并返回左孩子的返回值。

关于第上述第3点，为什么是先查找右孩子呢？因为我们要查的是最大值，而右孩子中所有元素的值都是大于左孩子的，因此先找右孩子，如果找失败了再找左孩子。

假设线段树的区间是[0,m]，可以证明这个查找的时间复杂度是O(logm)的，可惜这里空白处太小，写不下。

【算法流程】

由此可以得到算法的核心流程了。

\1.   对所有的白点和黑点以x升序排序。

\2.   从左到右遍历每个黑点，对于每个黑点，进行步骤3和4

\3.   将x值小于等于当前黑点，且未存入线段树的所有白点的y存入线段树。

\4.   设当前黑点的纵坐标为y_b，则去线段树中查找不大于y_b的最大的y。如果找到这样的y，那么在线段树中将y删除（cnt减1），并且匹配计数加1;如果没找到，说明匹配失败。



## 运树

**题目描述**

在一条直线上有n处地方，从左到右依次编号为1, 2, …, n，每处地方都有一定量的树木，每处地方的单位量树木运送单位距离都需要一定的费用，例如在1处有数量为a的树木，单位量的树木运送单位距离的费用是b，那么把这么多树木一共运送c的距离所需要的费用为a*b*c。现在需要把这n处地方的树木送往加工中心处理，为了使得路径单一，所有地方的树木只能向右边（编号较大的地方的方向）运送，已知第n处的地方已经存在了一个加工中心。为了减少树木运送的费用，在这条路径上最多可以添加k个加工中心，使得总运费最少。问最小的运费是多少？

**解答要求**时间限制：2000ms, 内存限制：100MB

**输入**

输入数据一共包括4行，第1行输入n和k(2<=n<=10000, 1<=k<=min(200, n-1))，表示n处地方，最多添加k个加工中心。第2行输入n个正整数，表示每个地方的树木总量，第3行输入n个正整数，表示每个地方单位数量单位距离运送的费用。第4行输入n-1个正整数，表示从左到右相邻两个地方的距离。除n以外，所有的数字都<=300。

**输出**

输出一行一个整数，表示最小的运费。

**样例**

输入样例 1 复制

```
3 1
1 2 3
3 2 1
2 2
```

输出样例 1

```
6
```

**提示**

Carry_tree简单题解

不难证明，加工厂只能建在节点上
比较容易想到状态方程：dp[k][n] = min{0<=i<n}(dp[k - 1][i] + delta(i + 1, n))
其中delta(i + 1, n)表示为从i + 1到n的所有树木运送到第n号节点的耗费总和
复杂度O(k*n*n)，会超时，需要优化。
采用斜率优化，我们令设几个变量。
sum[n]表示从1~n的所有的树木搬到n号节点所需要的代价总和
a[n]表示从1~n的所有的树木所搬运单位路程所需的代价
d[n]表示从1~n的距离
那么delta(i + 1, n) = sum[n] - sum[i] - a[i] * (d[n] - d[i])
所以原方程式化为dp[k][n] - sum[n] = min{0<=i<n}(dp[k - 1][i] - sum[i] + a[i] * d[i] - a[i] * d[n])
因为a[i]单调递增，因而采用斜率优化。

按字面意义，可设输入的2~4行为num[]、fee[]、dist[]，不难得出，“提示”中的对应变量为

a[0] = 0; a[i] = a[i-1] + num[i-1]*fee[i-1];

d[0] = d[1] = 0; d[i] = d[i-1] + dist[i-1];

sum[i] = 0; sum[i] = sum[i-1] + a[i-1]*dist[i-1];

由提示的目标函数可知本题为二维动态规划问题，dp[cur_k][i] 为总共 i 个节点时建立 cur_k 个厂的最小开销（i>=(cur_k+1)），易知其初始值 dp[0][i] = sum[i], (i∈[1, n])。二维遍历方式为内循环按 i 遍历（[cur_k+1: n]），外循环按 cur_k 遍历（[1: k]）。

由于推导式含有x[i]*y[i]形式的项，直接推导会超时间复杂度，且单调队列优化无效。这里采用斜率优化方法进行剪枝推导。

原式为 dp[cur_k][i] - sum[i] = min(j){dp[cur_k-1][j] - sum[j] - a[j]*(d[i] - d[j])}

我们这里固定i，若j是最小值点，则对任意m<j，都有 dp[cur_k-1][j] - sum[j] - a[j]*(d[i] - d[j]) <= dp[cur_k-1][m] - sum[m] - a[m]*(d[i] - d[m])



## 装箱子（未解）

```
题目描述
有大小不等的n个箱子排成一行，每个箱子大小为一个整数值，为了节约空间，需要将小箱子装入大箱子中，从而形成一个或多个大小为1~m逐层嵌套的箱子集合（最终的箱子集合中的箱子大小必须从1开始，且大小值连续，如[1,2,3,4,5]合法，而[1,2,4,5,6]或[2,3,4]则不合法）。
组合过程中，只能将位置相邻的箱子组合到一起，将一个箱子打开的过程称为1个单位成本，比如合并[1,2,7]与[5]，需要将大小为7的箱子打开拿出[1,2]，将大小为5的箱子打开放入[1,2]，然后将[1,2,5]放入7中，形成[1,2,5,7]，需要打开2次，则其组合成本为2，同理，合并[1,2,5]与[3,4]需要分别打开大小为5,3,4的箱子各1次，其组合成本为3。组合过程中不要求箱子集合内的箱子大小值连续。
求将n个箱子组合成一些箱子集合的最小成本。

解答要求
时间限制：1000ms, 内存限制：256MB
输入
第一行一个整数 n，表示待组合的箱子个数，第二行包含 n 个整数，依次表示每个箱子的大小值ai。

输出
如果答案存在则输出将n个箱子组装成箱子集合的最小成本，否则输出-1。

样例
输入样例 1 复制

8
1 2 3 1 2 4 3 3
输出样例 1

-1
提示样例 1
首先将1,2,3组合成一个集合[1,2,3]，然后将1,2,4,3给合成一个集合[1,2,3,4]，剩余一个3无法形成一个合法的集合，无解。



输入样例 2 复制

6
1 2 2 4 1 3
输出样例 2

6
提示样例 2
首先将1,2组合成一个集合[1,2]，需要打开大小为2的箱子1次，成本为1；
然后将2,4,1,3给合成一个集合[1,2,3,4]，需要先打开大小为4的箱子，放入2，成本1，形成[2,4]；
接下来打开大小为2、4的箱子各1次，放入1，成本2，形成[1,2,4]，
再打开大小为4的箱子，拿出[1,2]，打开大小为3的箱子，放入[1,2]，形成[1,2,3]，然后放入4，形成[1,2,3,4]，成本为2。
因此总成本为1+1+2+2=6。


提示
数据规模和约定: 1<=n(箱子数量)<=500 , 1<=ai(箱子大小)<=500

```

```python
def f():
    n = input()
    size = list(map(int, input().strip().split()))
    stack = []
    cur = []
    ret = 0
    for box in size:
        if not stack:
            stack.append(box)
        if box in stack:
            if cur:
                _new_stack = []
                for _tmp_box in stack:
                    if _tmp_box < cur[-1]:
                        _new_stack.append(_tmp_box)
                    else:
                        _new_stack.extend(cur)
                        _new_stack.append(_tmp_box)
                        ret += 1

                if _new_stack[0] != cur[0]:
                    ret += 1
                stack = _new_stack

            if stack[0] != 1:
                print(-1)
                return
            if len(stack) > 1:
                _tmp_stack = sum([stack[i+1] - stack[i] for i in range(len(stack)-1)])
                if _tmp_stack != len(stack) - 1:
                    print(-1)
                    return
            stack = [box]
        else:
            if not cur:
                cur.append(box)
            elif abs(box - cur[-1]) == 1:
                cur.append(box)
                ret += 1
                cur.sort()
            else:
                # 将cur放入stack
                if stack[-1] < cur[-1]:
                    ret += 1
                    stack.extend(cur)
                else:
                    _new_stack = []
                    for _tmp_box in stack:
                        if _tmp_box < box:
                            _new_stack.append(_tmp_box)
                        else:
                            _new_stack.extend(cur)
                            _new_stack.append(_tmp_box)
                            ret += 1

                    if _new_stack[0] != cur[0]:
                        ret += 1
                    stack = _new_stack

    if stack[0] != 1:
        print(-1)
        return
    if len(stack) > 1:
        _tmp_stack = sum([stack[i+1] - stack[i] for i in range(len(stack)-1)])
        if _tmp_stack != len(stack) - 1:
            print(-1)
            return
    print(ret)
```



## 办公楼改造（未解）

```
题目描述
溪流背坡村自2018年6月起陆续投入使用，12个欧洲经典建筑群，为员工提供了舒适健康的办公环境。参考华为坂田基地，后续在某一时间可能需要对所有办公楼进行翻新改造。翻新改造可能使它的容量发生改变。为了改造，需要新建办公楼以便改造期间员工正常办公。当然，新建办公楼越小越省钱。翻新改造的原则参考：
1、可以按照任意顺序翻新改造；
2、翻新改造前，需要把该栋楼上所有员工搬移到一个或者多个其他楼栋；
3、翻新改造后，该栋楼可以立刻开始使用；
4、没有必要把员工放到它原来的办公楼栋；（不做限制）
5、员工也可以搬到新建的办公楼栋；
举个例子，假设有4个楼栋A、B、C、D，容量分别为6、1、3、3（单位：100人），翻新改造后，它的容量分别为6、7、5、5（单位：100人）。
如果只新建容量为1（单位：100人）的办公楼，可以把B栋的员工搬到新建办公楼，然后翻新改造B栋，改造后B栋的容量是7（单位：100人）了，那么你就可以把A栋的员工搬过去然后翻新改造A栋，最后把C、D栋员工搬迁到A上，再翻新改造C栋和D栋。

解答要求
时间限制：1000ms, 内存限制：256MB
输入
第一行一个数n（1≤n≤1,000,000），表示办公楼个数。接下来n行，每行两个数a和b，分别表示该办公楼的原容量和翻新改造后的容量。所有容量都以100人为单位，且1≤a,b≤1,000,000,000。

输出
如果要翻新改造所有办公楼，最少需要新建容量多大办公楼（单位：100人）。

样例
输入样例 1 复制

10
11 82
98 12
78 53
15 10
41 2
81 58
53 42
30 41
25 39
20 54
输出样例 1

61
提示样例 1


提示
```



## 缓存技术（未解）

```
题目描述
在计算机中，CPU 只能和高速缓存 Cache 直接交换数据。当所需的内存单元不在 Cache 中时，则需要从主存里把数据调入 Cache。此时，如果 Cache 容量已满，则必须先从中删除一个。
例如，当前 Cache 容量为 3，且已经有编号为 10 和 20 的主存单元。
此时，CPU 访问编号为 10 的主存单元，Cache命中。
接着，CPU 访问编号为 21 的主存单元，那么只需将该主存单元移入 Cache 中，造成一次缺失（Cache Miss）。
接着，CPU 访问编号为 31 的主存单元，则必须从 Cache 中换出一块，才能将编号为 31 的主存单元移入 Cache，假设我们移出了编号为 10 的主存单元。
接着，CPU 再次访问编号为 10 的主存单元，则又引起了一次缺失。我们看到，如果在上一次删除时，删除其他的单元，则可以避免本次访问的缺失。
在现代计算机中，往往采用 LRU（最近最少使用）的算法来进行 Cache 调度——可是，从上一个例子就能看出，这并不是最优的算法。
对于一个固定容量的空 Cache 和连续的若干主存访问请求，聪聪想知道如何在每次 Cache 缺失时换出正确的主存单元，以达到最少的 Cache 缺失次数。

解答要求
时间限制：1000ms, 内存限制：256MB
输入
输入文件第一行包含两个整数 N 和 M，分别代表了主存访问的次数和 Cache 的容量。
第二行包含了 N 个空格分开的正整数，按访问请求先后顺序给出了每个主存块的编号。

输出
输出一行，为 Cache 缺失次数的最小值。

样例
输入样例 1 复制

6 2
1 2 3 1 2 3
输出样例 1

4
提示样例 1
```



## 踏春（未解）

```
题目描述
2022年4月，深莞疫情结束。松研产品部A和解决方案部B两个部门组织联合环湖自行车活动，同时，松山湖风景区为提升人气，推出双人自行车免费活动。A和B两个部门非常大，只有部分同事相互比较熟悉，只有两个来自不同部门且相互熟悉的人一起骑双人自行车两人才开心。
1、假定松山湖只有单人和双人两种车型且车数量不限，优惠后单人自行车租金10元/次，双人自行车免费（双人自行车不允许当做单人自行车使用）；
2、活动必须保证每个人都有车骑，骑双人车的务必保证两人都开心（来自不同部门，且相互熟悉）。
作为软件高手，请你帮活动组织者计算，如何安排才能使租车费用最低。

解答要求
时间限制：1000ms, 内存限制：256MB
输入
第一行包含两个整数，N (0 <= N <= 200) 和 M (0 <= M <= 200)，分别表示A和B两个部门参加活动的人数，两部门的人各自分开编号，编号从1开始。
从第二行开始，每一行依次代表A部门的员工Ai的熟人关系，参考提示样例，以下 N 行中的第一个整数 (SumAi) 指示后续Bj（1 <= Bj <= M）的总数；

输出
输出带有单个整数的单行，表示租车的最低费用。

样例
输入样例 1 复制

2 3
2 1 3
1 1
输出样例 1

10
提示样例 1
2 3 // 表示N=2,M=3，A部门有2个人、B部门有3个人参加活动
2 1 3 // A1有2个熟人，分别是B1、B3
1 1 // A2有1个熟人，是B1
一种费用最低的策略是A1-B3、A2-B1两辆双人车，B2单人车，这样只需要一辆单车费用
```



## 质数

```
题目描述
HASH是基站软件系统中常用的一种的查找算法，减少HASH冲突是算法性能的关键因素之一。为减少冲突，可以将待匹配项除以一个质数作为KEY来散列。质数是指在大于1的自然数中，除了1和它本身以外不再有其他因数的自然数。 质数又称素数。质因数（素因数或质因子）在数论里是指能整除给定正整数的质数。
给定一个正整数，请找出它是否是质数。

解答要求
时间限制：1000ms, 内存限制：256MB
输入
第一行包含测试用例的数量 T (1 <= T <= 20 )，然后接下来的 T 行每行包含一个整数 N (2 <= N <= 2^44 )。

输出
对于每个测试用例，如果 N 是素数，则输出一行包含单词“Prime”，否则，输出一行包含 N 的最小素因数。

样例
输入样例 1 复制

2
5
10
输出样例 1

Prime
2
提示样例 1
```

```python
import math

def f():
    T = int(input())
    data = []
    for _idx in range(T):
        data.append(int(input()))
    for n in data:
        if n == 2 or n == 3:
            print("Prime")
            continue
        if n % 2 == 0:
            print(2)
            continue
        if n % 3 == 0:
            print(3)
            continue
        else:
            m = int(math.sqrt(n)) + 1
            for i in range(5, m, 6):
                if n % i == 0:
                    print(i)
                    break
            else:
                for j in range(7, m, 6):
                    if n % j == 0:
                        print(j)
                        break
                else:
                    print("Prime")
            # return True


if __name__ == "__main__":
    import time
    max_prime = 17592186044399
    t1 = time.time()
    print(f(max_prime))
    # for i in range(2, int(math.sqrt(n))+1):
    #     if n % i == 0:
    #         break
    # else:
    #     print("Prime")
    t = time.time() - t1
    print(t)

    # print(4194301 * 4194301)
    # f()
```

## 共享单车管理系统

仓库（根）存放或提供不限量单车

租借点：租赁或归还单车，有限容量

输入：[-1,0,1,1,5,0,1,0], 41， 列表元素索引即租借点编号，索引0为仓库，后续元素为租借点，值表示父节点的编号。最后一个数字租借点capacity容量。

初始化：初始租借点有[capacity/2]量共享单车

rentbike：数量不足向父节点索取  returnbike：容量不足向父节点归还

```python
import time
from typing import List
from collections import deque


class Node:
    def __init__(self, index, obj_parent, capacity, bike_num):
        self.index = index
        self.son_list = []
        self.capacity = capacity
        self.bike_num = bike_num
        self.parent = obj_parent

    def dfs(self, index):
        visited, stack = set(), deque([self])
        while stack:
            vertex = stack.pop()
            if vertex.index == index:
                return vertex
            for son in vertex.son_list:
                if son not in visited:
                    visited.add(son)
                    stack.append(son)
        return None

    def bfs(self, index):
        queue = deque([self])
        while queue:
            vertex = queue.popleft()
            if vertex.index == index:
                return vertex
            # 树不存在环, 可以省略访问记录
            queue.extend(vertex.son_list)
        return None

    def add(self, obj_nd):
        obj_parent = self.bfs(obj_nd.parent.index)
        if obj_parent:
            obj_parent.son_list.append(obj_nd)


# 原题目中描述可用树表示，实际上不需要做树结构，用链表即可
class BikeSharingSystem_2:
    def __init__(self, pre_node: List[int], capacity: int):
        self.ini_bike_num = int(capacity/2)
        self.capacity = capacity
        self.nodes = {i: Node(i, None, capacity, self.ini_bike_num) for i in range(len(pre_node))}
        i = 1
        for j in pre_node[1:]:
            self.nodes[i].parent = self.nodes[j]
            i += 1

    def rent_bikes(self, node_id: int, num: int) -> int:
        rec_node = cur_node = self.nodes.get(node_id)
        while num > 0:
            if rec_node.bike_num >= num:
                rec_node.bike_num -= num
                num = 0
            else:
                if rec_node.parent:
                    num -= rec_node.bike_num
                    rec_node.bike_num = 0
                    rec_node = rec_node.parent
                else:
                    rec_node.bike_num = 0
        return cur_node.bike_num

    def return_bikes(self, node_id: int, num: int) -> int:
        rec_node = cur_node = self.nodes.get(node_id)
        while num > 0:
            space = rec_node.capacity - rec_node.bike_num
            if space >= num:
                rec_node.bike_num += num
                num = 0
            else:
                if rec_node.parent:
                    num -= space
                    rec_node.bike_num = self.capacity
                    rec_node = rec_node.parent
                else:
                    rec_node.bike_num = self.capacity
                    num = 0
        return cur_node.bike_num

    def reset(self) -> int:
        cnt = 0
        for i, nd in self.nodes.items():
            if nd.parent:
                if nd.bike_num == 0 or nd.bike_num == self.capacity:
                    nd.bike_num = self.ini_bike_num
                    cnt += 1
            else:
                nd.bike_num = self.ini_bike_num
        return cnt

    def get_top5_nodes(self) -> List[int]:
        nums = [(nd.bike_num, i) for i, nd in self.nodes.items() if i != 0]
        nums.sort(key=lambda nums: (nums[0], -nums[1]), reverse=True)
        nums = [(index, num) for num, index in nums]
        return [index for index, num in nums[:5]]


# 下面是树的思路
class BikeSharingSystem:

    def __init__(self, pre_node: List[int], capacity: int):

        bike_num = int(capacity/2)

        self.capacity = capacity
        self.root = Node(0, None, 99999999 * 2, 99999999)
        self.all_nodes = []

        pre_node = [(i, pre_node[i]) for i in range(1, len(pre_node))]

        while pre_node:
            i = 0
            registered = set()

            for index, parent_index in pre_node:
                obj_parent = self.root.bfs(parent_index)
                if obj_parent:
                    new_node = Node(index, obj_parent, capacity, bike_num)
                    self.all_nodes.append(new_node)
                    obj_parent.add(new_node)
                    registered.add(i)
                i += 1

            registered = list(registered)
            while registered:
                pre_node.pop(registered.pop())

    def rent_bikes(self, node_id: int, num: int) -> int:
        obj_node = self.root.bfs(node_id)

        if obj_node:
            if obj_node.bike_num >= num:
                obj_node.bike_num -= num
            else:
                if obj_node.parent:
                    self.rent_bikes(obj_node.parent.index, num - obj_node.bike_num)
                    obj_node.bike_num = 0
                else:
                    raise RuntimeError('Not enough bike.')
            return obj_node.bike_num

    def return_bikes(self, node_id: int, num: int) -> int:
        obj_node = self.root.bfs(node_id)

        if obj_node:
            space = obj_node.capacity - obj_node.bike_num
            if space >= num:
                obj_node.bike_num += num
            else:
                if obj_node.parent:
                    self.return_bikes(obj_node.parent.index, num - space)
                    obj_node.bike_num = obj_node.capacity
            return obj_node.bike_num

    def reset(self) -> int:
        self.root.capacity = 99999999
        self.root.bike_num = 9999
        cnt = 0
        for nd in self.all_nodes:
            if nd.bike_num == 0 or nd.bike_num == self.capacity:
                nd.bike_num = int(self.capacity / 2)
                cnt += 1
        return cnt

    def get_top5_nodes(self) -> List[int]:
        nums = [(nd.bike_num, nd.index) for nd in self.all_nodes]
        nums.sort(reverse=True)
        nums = [(index, num) for num, index in nums]

        start = 0
        num = nums[start][1]
        for i in range(1, len(nums)):
            if nums[i][1] != num:
                nums[start: i] = sorted(nums[start: i])
                start = i
                num = nums[start][1]
        nums[start:] = sorted(nums[start:])

        return [index for index, num in nums[:5]]


if __name__ == "__main__":
    t0 = time.time()
    for i in range(100000):
        bike_namager = BikeSharingSystem([-1,0,1,1,5,0,1,0,3,4,4,2,6,7,5,7,8,6], 41)
        bike_namager.rent_bikes(2, 31)
        bike_namager.rent_bikes(3, 45)
        bike_namager.get_top5_nodes()
        bike_namager.return_bikes(5, 29)
        bike_namager.return_bikes(5, 100)
        bike_namager.reset()
        bike_namager.rent_bikes(3, 12)
        bike_namager.get_top5_nodes()
    print('time:{}'.format(time.time()-t0))

    t0 = time.time()
    for i in range(100000):
        bike_namager2 = BikeSharingSystem([-1,0,1,1,5,0,1,0], 41)
        bike_namager2.rent_bikes(2, 31)
        bike_namager2.rent_bikes(3, 45)
        bike_namager2.get_top5_nodes()
        bike_namager2.return_bikes(5, 29)
        bike_namager2.return_bikes(5, 100)
        bike_namager2.reset()
        bike_namager2.rent_bikes(3, 12)
        bike_namager2.get_top5_nodes()
    print('time:{}'.format(time.time()-t0))
```



## 订票系统

每interval天举办一次技术讲座，第k次讲座在k*interval天举办，开发订票时间是讲座前的interval天以内 [(k-1)*interval, k*interval-1]

每次讲座有100*100个座位，行列号0开始编号

订单列表先到先得： orders[i] = [day, row0, col0, row1, col1, ...rowM, colM]表示，订票时间，按优先级排序的多个座位，预订第一个成功的座位

返回多少个定点预订成功

```python
# 思路，将天数和座位组合成一个编号，检测不重复编号+1

from typing import List


def get_successful_num(interval: int, orders: List[List[int]]) -> int:
    ans = 0
    m = list()
    for order in orders:
        n = len(order) // 2
        k = order[0] // interval
        for i in range(n):
            row = order[i * 2 + 1]
            col = order[i * 2 + 2]
            s = k * 1000000 + row * 100 + col
            if m is None:
                ans += 1
                m.append(s)
            else:
                if s not in m:
                    ans += 1
                    m.append(s)
                    break
    return ans


interval = 3
orders = [[0, 1, 1], [0, 0, 2], [1, 1, 1], [1, 0, 2, 0, 0], [3, 0, 2]]

print(get_successful_num(interval, orders))
```

# Basic Algrithem

## 欧拉函数

功能：用于求小于n的与n互质数的个数

```python
import math

def euler_function_1(n):
    ret = 1
    for i in range(2, math.ceil(math.sqrt(n))):
        if n % i == 0:
            n /= i
            ret *= i-1
            while n % i == 0:
                n /= i
                ret *= i
    if n > 1:
        ret *= n-1
    return ret


def euler_function_2(n):
    j = 1
    for i in range(2, n):
        a = n
        q = i
        r = a % i
        while r != 1 and r != 0:
            a = int(q)
            q = int(r)
            r = int(a) % int(q)
        # 如果互素则计数+1
        if r == 1:
            j += 1
        elif r == 0:
            pass
    print(j)


if __name__ == '__main__':
    print(euler_function_1(int(input('请输入n:'))))




int p[maxn];
void phi()
{
    me(p,0);
    p[1]=1;
    for(int i=2;i<maxn;i++)
    {
        if(!p[i])
            for(int j=i;j<=maxn;j+=i)
            {
                if(!p[j])p[j]=j;
                p[j]=p[j]/i*(i-1);
            }
    }
}
```





## 递归（汉诺塔）

功能：三根柱子分别为A，B，C，打印汉诺塔的移动过程，典型的递归函数

输入：n，饼的个数

输出：打印汉诺塔的移动过程

```python
def hannoi_move(n, a, b, c):
    if n == 1:
        print('%s to %s' % (a, c))
    else:
        hannoi_move(n-1, a, c, b)
        hannoi_move(1, a, b, c)
        hannoi_move(n-1, b, a, c)
hannoi_move(3, 'A', 'B', 'C')
```



## Pollard rho算法

**Pollard-Rho算法**是John Pollard发明的一种能快速找到大整数的一个非1、非自身的因子的算法。

https://zhuanlan.zhihu.com/p/267884783

https://www.cnblogs.com/812-xiao-wen/p/10544546.html



## 平衡二叉树

数据结构之树：https://zhuanlan.zhihu.com/p/35035166

https://blog.csdn.net/weixin_45666566/article/details/108092977

https://blog.caref.xyz/computer%20tech/binary-search-tree/
