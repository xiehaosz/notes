

https://www.luogu.com.cn/problem/list?difficulty=5&page=1

Leetcode

https://oi-wiki.org/dp/memo/

# Simple

## 打印表格

**题目：**使用”+”和”-”两个符号打印一个n×m的表格，（水平方向有三个”-”，竖直方向有一个”| ”，”|”对齐”+”）的矩形表格为

**输入：**输入只有一行包含两个整数n和m(0<n,m<13)

**输出：**输出n×m的表格

**样例：**1 1 

```
+---+
|   |
+---+
```

```python
def func():
    # please define the python3 input here. 
    # For example: a,b = map(int, input().strip().split())
    n, m = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    row_init = "+---+"  # 初始化的图形
    col_init = "|   |"  # 初始化的图形
    row_add = "---+"  # 叠加图形
    col_add = "   |"  # 叠加图形
    print(row_init + row_add*(m-1))  # 输出图形基础
    for i in range(n):  # 根据行进行叠加
        print(col_init + col_add*(m-1))  # 根据列进行叠加
        print(row_init + row_add*(m-1))  # 根据列进行叠加
if __name__ == "__main__":
    func()
```

```python
def func():
    # please define the python3 input here.
    # For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    a, b = map(int, input().strip().split())
    res = '\n'.join([''.join(['+' if i % 4 == 0 else '-' for i in range(4 * b + 1)]) if j % 2 == 0 else ''.join(
        ['|' if i % 4 == 0 else ' ' for i in range(4 * b + 1)]) for j in range(2 * a + 1)])
    print(res)
```



## 加法

输入：2个数字，直到结束符号

输出：和

```python
def func():
    while True:
        try:
            a, b = map(int, raw_input().strip().split())
            print a + b
        except EOFError:
            break
if __name__ == "__main__":
    func()
```



# Medium

## N进制小数

**输入：**输入包含两个数m,n，用空格隔开。输入包含多组测试，当m,n都为0时输入结束

**输出：**输出10进制正小数m的n进制小数。结果保留10位小数

```python
# 执行用时12ms
def dec_decimal_transform(m, n, digit_num=5):
    digit_n = []
    for i in range(digit_num):
        d = m * n
        q = int(d)
        m = d - q
        digit_n.append(q)
    return '%.10f' % float('0.' + ''.join(map(str, digit_n)))

if __name__ == "__main__":
    while 1:
        m, n = str_in = input().strip().split()
        if m == n == '0':
            break
        else:
            print(dec_decimal_transform(float(m), int(n) , digit_num=10))
```

```python
# 执行用时4ms
import math
def func():
    while True:
        try:
            str = input()
            strlist = str.split(" ")
            a,b = float(strlist[0]),int(strlist[1])
            #print(a,b)
            if b !=0 :
                aaa = 0
                for i in range(1,11):
                    num = int(a*b)
                    aaa += num *0.1 ** i
                    a = math.modf(a*b)[0]
                print("%.10f" % aaa)
        except EOFError:
            break
                
if __name__ == "__main__":
    func()
```

## 最小公倍数

输入：输入一个整数n,(1<n<=100），测试包含组样例，**读到文件末尾结束**。

输出：数字1至n的最小公倍数

```python
def min_prx(m, n):
    """最小公倍数=两树乘积除以最大公约数"""
    temp = m * n
    if temp == 1:
        return 1
    if m < n:
        return min_prx(n, m)
    while n != 0:
        m, n = n, m % n
    return int(temp / m)


if __name__ == "__main__":
    while True:
        try:
            n = int(input())
            res = 1
            for i in range(1, n + 1):
                res = min_prx(res, i)
            print(res)
        except:
            break
```

```python
if __name__ == "__main__":
"""每个数如果被更小的数整除，则在list中记录被除数"""
        while True:
            try:
                n = int(input())
                l = list(range(1, n+1))
                for i in range(n):
                    for j in range(i):
                        if l[i] % l[j] == 0:
                            l[i] = int(l[i]/l[j])
                tmp = 1
                for i in range(n):
                    tmp *= l[i]
                print(tmp)
            except:
                break
```

## 最大公约数 Plus

**题目：**

从n个不同元素中，任取m(m≤n)个元素并成一组，叫做从n个不同元素中取出m个元素的一个组合；从n个不同元素中取出m(m≤n)个元素的所有组合的个数，叫做从n个不同元素中取出m个元素的组合数，用符号c(n,m)表示，计算公式为：c(n,m)=n!/((n-m)!×m!)

现在你的任务是求出C(2n,1),C(2n,3),C(2n,5),…,C(2n,2n-1)的最大公约数。

**输入：**一个整数n(1<n<=10000)

**输出：**C(2n,1),C(2n,3),C(2n,5),…,C(2n,2n-1)的最大公约数

```python
''' 根据输入输出去猜测关系（可以观察到关系，但是不严谨，需要证明）
(in:out)(2:4)(3:2)(4:8)(5:2)(6:4)(7:2)(8:16)(10:4)(11:2)(12:8)(13:2)(14:4)(15:2)(16:32)
关系猜测：对输入值因子分解，如果因子中含有n个2，则输出为2 ** (n+1)'''
while True:
    try:
        n = int(input())
        count = 1
        while n % 2 == 0:
            n = n / 2
            count += 1
        print(2**count)
    except:
        break
```

```python
def func(n):
    result = 2 * max(n, 2 ** (2 * n - 2))
    print(result)
def max(a,b):
    while a%b!=0:
        a,b=b,(a%b)
    return b
if __name__ == "__main__":
    n = int(input())
    func(n)
```

```python
n=int(input().strip())
print((2*n)&(-2*n))
```



## 螺旋队列

```
21 22……
20  7  8  9 10
19  6  1  2 11
18  5  4  3 12
17 16 15 14 13
```

**题目：**看清楚以上数字排列的规律，设1点坐标为(0,0)，x方向向右为正，y方向向下为正。例如7的坐标为(-1,-1)，2的坐标为(1,0)，3的坐标为(1,1)。
编程实现输入任意坐标(x,y)，输出对应的数字

**输入：**输入包含多组测试，每组测试占一行，包含两个整数x,y , (-100 ≤ x, y ≤ 100), **输入到文件末尾结束**

**输出**：输出对应的数字，每组测试占一行

```python
def func():
    a="a"
    while a=="a":
        try:
            x,y=map(int,input().split()) # x,y=list(map(lambda x:int(x),input().split(" "))) 
            if x==0 and y==0:
                print(1)
            else:
                if abs(x)>abs(y):
                    print(int(4*(x**2)-2*x-abs(x)+1+x/abs(x)*y))
                else:
                    print(int(4*(y**2)-2*y+abs(y)+1-y/abs(y)*x))
        except Exception as Ex:
            break
```

```python
def luoxuan(x, y):
    k = 0
    if abs(x) >= abs(y):
        if x >= 0:
            if x == -y:
                return (2*x+1)**2
            else:
                return (2 * x - 1) ** 2 + x + y
        else:
            if x == y:
                return (-2*x+1)**2+2*x
            else:
                return (-2 * x - 1) ** 2 - 5 * x - y
    else:
        if y < 0:
            return (-2 * y + 1) ** 2 + y + x
        else:
            return (2 * y + 1) ** 2 - 5 * y - x
def func():
    while True:
        try:
            x, y = map(int, input().strip().split())
            print(luoxuan(x, y))
        except EOFError:
            break
        except ValueError:
            break
```





## Olympic Game

**题目：**

奖牌榜的排名规则如下：

1. 首先gold medal数量多的排在前面
2. 其次silver medal数量多的排在前面
3. 然后bronze medal数量多的排在前面
4. 若以上三个条件仍无法区分名次，则以国家名称的字典序排定。

**输入：**第一行输入一个整数N(0<N<21)，代表国家数量。接下来的N行，每行包含一个字符串Namei表示每个国家的名称，和三个整数Gi、Si、Bi表示每个获得的gold medal、silver medal、bronze medal的数量，以空格隔开，如(China 51 20 21)。

**输出：**输出奖牌榜的依次顺序，只输出国家名称，各占一行

```python
def func():
    country = int(input().strip())
    res = [input().strip().split() for i in range(country)]
    result = sorted(res, key=lambda x:(-int(x[1]), -int(x[2]), -int(x[3]), x[0]))
    print('\n'.join([r[0] for r in result]))
if __name__ == "__main__":
    func()
```

```python
# 关键：对金银铜牌数量存负数，用lambda函数定义比较顺序
def func():
    n = int(input())
    nums = []
    for _ in range(n):
        name, g, s, b = input().split()
        nums.append((-int(g), -int(s), -int(b), name))
    nums.sort()
    for line in [x[3] for x in nums]:
        print(line)
if __name__ == "__main__":
    func()
```

```python
# 排序国家>排序铜牌>排序银牌>排序金牌
while 1:
    try:
        n = int(input())
        ret = []
        for i in range(n):
            a, b, c, d = input().split()
            b, c, d = int(b), int(c), int(d)
            ret.append((a, b, c, d))
        ret = sorted(ret, key=lambda x:x[0])
        s1 = sorted(ret, key=lambda x:x[3], reverse=True)
        s2 = sorted(s1, key=lambda x:x[2], reverse=True)
        s3 = sorted(s2, key=lambda x:x[1], reverse=True)
        for i in range(n):
            print(s3[i][0])
    except:
        break
```

## Prime Plus

**输入：**Input only has an integer n (0<n<100000001) in a line

**输出：**the number of prime numbers which are less than or equal to n

思路：

1. 暴力解法:遍历小于n的数校验是否为素数
2. 过滤偶数:遍历小于n的数校验是否为素数
3. 一个数的因子不可能大于它的平方根，除数范围缩小至[2,√n+1]
4. 假如n是合数，必然存在非1的两个约数p1和p2，其中p1<=sqrt(n)，p2>=sqrt(n)。由此我们可以改进上述方法优化循环次数
5. 数据学归纳：假设有6x分别加1,2,3,4,5，6x+2or4和6x+3分别被2,3整除，所以质数总是等于 6x-1(即6x+5)或者 6x+1，其中 x 是大于等于1的自然数。代码循环步长可以为6

```python
import math
def func_ai(n):
    # 埃氏筛法：素数倍数不为素数。构造一个初值为1，大小为n的素数标志列表。每发现一个素数就把所有它的倍数对应标志都置为0。
	prime_list = bytearray((True for _ in range(n + 1))) 
    '''bytearray将传入的数组(元素数值范围0~255)转换为一个字节数组，而复制和访问复杂度和列表无异, 但可以做到8倍的内存压缩'''
	prime_list[0] = prime_list[1] = False  # 0,1特殊处理
	for i in range(2, n+1):
		if not prime_list[i]:
			continue
		for j in range(i * 2, n + 1, i):
			prime_list[j] = False
	print(sum(prime_list))
```

```python
def func_eur(n):
	# 欧拉筛法：任意一个合数都为一个素数和一个数的乘积, 让每个合数只被自己最小的质因数标记
	primes_bool_list = bytearray((1 for _ in range(n+1)))
	primes_list = list()
	for index_bool in range(2, n + 1):
		if primes_bool_list[index_bool] == 1:
    		primes_list.append(index_bool)
		for prime in primes_list:
			if index_bool * prime > n:
				break
             primes_bool_list[index_bool * prime] = 0
			if index_bool % prime == 0:
				break
	print(len(primes_list))
```



## Vowel

输入：输入一个字符串S(长度不超过100，只包含大小写的英文字母和空格)

输入：将元音字母写成大写的，辅音字母则都写成小写

```python
if __name__ == "__main__":
    s = input().strip().lower()
    for i in 'aeiou':
        s = s.replace(i, i.upper())
    print(s)
```

```python
# 低效
s = ''.join([x if x not in 'aoeiu' else x.upper() for x in input().strip().lower()])
print(s)
```

## 超级计算器

输入：输入一个包含四则运算、小括号的表达式，如：4+2，或((((2+3)*2+4.02)*4)+2)*4。**输入包含多组数据**，我们确保没有非法表达式。当一行中只有0时输入结束，相应的结果不要输出。

输出：对每个测试用例输出1行，即该表达式的值，精确到小数点后2位，如输入是1+2+3，则输出6.00。

样例：

```
-2
5.6*(-2*(1+(-3)))
2*((4+2)*5)-7/11
1+2+3
0
-2.00
22.40
59.36
6.00
```





## Calculator Ver.2

题目：编写一个程序来检验算式的括号是否完全匹配

输入：输入只有一行，即一个长度不超过100的字符串S，表示Solo的算术表达式，（你只需考虑相互之间的括号是否完全匹配，不需考虑表达式的其他合法问题）。注意：S中不一定包含括号。

输出：若表达式的括号完全匹配了则输出“YES”，否则输出“NO”

样例：5.6*(-2*(1+(-3)))，YES， 注意(1+2)(23)是括号完全匹配的，((1+2)(23)和((1+2)23则没有完全匹配。

```python
# 投机：使用自带eval函数，能计算出结果证明表达式是正确的，抛异常了说明是错误的
import sys
try:
    input_str = str(sys.stdin.readline().strip())
    eval(input_str)
    print('YES')
except Exception as e:
    print('NO')
```

```python
'''思路：统计左括号和右括号的数量，当他们相等时才有可能是正确的格式（特殊情况是不带括号的，这里单独判断），然后括号数量相等后再用正则表达式进行匹配过滤一遍，匹配左括号开头右括号结尾的最长字符串，过滤匹配之后的字符串要是格式正确的话，它的左括号和右括号的数量和匹配前必然是相等的，其它错误格式匹配出来的字符串括号数都会减小，所有的组合包括以下四种情况：
string1 = “(1+1)+(2+2)” 正确
string2 = “)1+1+(1+2)*2(“ 错误
string3 = “)-1(+1+3+(4-1)”错误
string4 = “(4-3)+)-4-5( “错误
'''
import re
def func():
    while True:
        try:
            input_string = input()
            if input_string.count("(") == 0 and input_string.count(")") == 0:
                print("YES")
            elif input_string.count("(") != input_string.count(")"):
                print("NO")
            else:
                filter_string = re.findall("\(.+\)",input_string)[0]
                if filter_string.count("(") == input_string.count("(") and filter_string.count(")") == input_string.count(")"):
                    print("YES")
                else:
                    print("NO")
        except:
            break
if __name__ == "__main__":
    func()
```



# HARD

## **Standings**

**输入：**整数N(0<N<109)

**输出：**可以整除N的所有正整数

**思路：**关于这道题目前没有找到更好的方法找出所有的因子。所以我们得用loop一个一个测试。这样的time complexity 会变成O(n)。不过我们可以用一个小方法把time complextiy 变成O（logn）。我们只需要把loop的上限从n改成n的平方根。这样也会得出正确的结果是因为一个数的因子不可能大于它的平方根。我们在loop的每一步验证n是否可以整除i。 如果可以就把他加到我们的备胎池子里。

```python
import math
def func(n):
    rs = set()
    for i in range(1, math.ceil(math.sqrt(n))+1):
        if not n % i:
            rs.add(i)
            rs.add(n//i)
    rs = sorted(list(rs))
    print(len(rs), end ="")
    for c in rs:
        print(" " + str(c), end ="")
if __name__ == "__main__":
    n = int(input().strip())
    func(n)
```

## **Word Maze**

**题目：**Word Maze 是一个网络小游戏，你需要找到以字母标注的食物，但要求以给定单词字母的顺序吃掉。假设给定单词if，你必须先吃掉i然后才能吃掉f。
但现在你的任务可没有这么简单，你现在处于一个迷宫Maze（n×m的矩阵）当中，里面到处都是以字母标注的食物，但你只能吃掉能连成给定单词W的食物。
注意区分英文字母大小写,并且你只能上下左右行走。

**输入：**输入第一行包含两个整数n、m(0<n,m<21)分别表示n行m列的矩阵，第二行是长度不超过100的单词W，从第3行到第n+2行是只包含大小写英文字母的长度为m的字符串。

**输出：**如果能在地图中连成给定的单词，则输出“YES”，否则输出“NO”。注意：每个字母只能用一次

**思路：**DFS算法，模拟递归

```python
# DFS解法
def func():
    # please finish the function body here.
    n, m = map(int, input().strip().split())
    word = input().strip()
    board = []
    for i in range(n):
        temp_str = input().strip()
        temp=[]
        for j in range(m):
            temp.append(temp_str[j])
        board.append(temp)
    def dfs(i, j, k):  # DFS定义i为走到了第几行，j为走到了第几列，k为当前判断到哪一位字符
        if not 0<=i<n or not 0<=j<m or word[k] != board[i][j]:  # 边界条件、字符不等的情况判定，如果不符合条件直接返回
            return False
        if k == len(word) - 1:  # 如果已经判断完了单词的最后一位，那就证明路径存在，则返回True
            return True
        tmp, board[i][j] = board[i][j], '/'  # 标记已经走过的路径
        res = dfs(i-1, j, k+1) or dfs(i+1, j, k+1) or dfs(i, j+1, k+1) or dfs(i, j-1, k+1)  # 上下左右寻找可行走的路径，并且对单词的下一位进行判断
        board[i][j] = tmp  # 不满足条件后回溯，其实很像二叉树的遍历，感兴趣的可以了解
        return res  # 返回得到的结果
    for i in range(n):  # 从第一行开始遍历
        for j in range(m):  # 从第一列开始遍历
            if dfs(i, j, 0):   # 如果res存在为True的，则直接返回并且输出YES
                print('YES')
                return True
    print('NO')  # 不存在True，则证明没有合适的路径，打印NO
    return False
if __name__ == "__main__":
    func()
```

```python
# If you need to import additional packages or classes, please import here.
def find_str(target_str, visited_coordinates, visited_str, dims, maps,
             row_count, col_count):
    # 当前字符没搜索过就继续
    if dims not in visited_coordinates:
        point_str = maps[dims[0]][dims[1]]
        # 将该坐标加入到已搜索的列表中
        visited_coordinates.append(dims)
        # 将该字符串加入已搜索的列表中
        visited_str.append(point_str)
        # 如果加入当前字符串之后字符串与目标一致，搜索结束
        if target_str == "".join(visited_str):
            print("YES")
            exit(0)
        # 如果加入当前字符串之后，是目标字符串子集，那么继续搜索
        elif target_str.startswith("".join(visited_str)):
            # 搜索方式是以当前坐标点为最新的坐标，进行四个方向的检索
            x, y = dims
            if x - 1 >= 0:
                left = [x - 1, y]
                find_str(target_str, visited_coordinates, visited_str, left,
                         maps, row_count, col_count)
            if x + 1 < row_count:
                right = [x + 1, y]
                find_str(target_str, visited_coordinates, visited_str, right,
                         maps, row_count, col_count)
            if y - 1 >= 0:
                up = [x, y - 1]
                find_str(target_str, visited_coordinates, visited_str, up,
                         maps, row_count, col_count)
            if y + 1 < col_count:
                down = [x, y + 1]
                find_str(target_str, visited_coordinates, visited_str, down,
                         maps, row_count, col_count)
        # 搜索完毕未找到，去除当前坐标，返回上一个坐标继续检索
        visited_coordinates.pop()
        visited_str.pop()
def process(input_dims, target_str, input_str):
     # 存放坐标对应的字符
    maps = []
    for line in input_str.split("\n"):
        maps.append([char for char in line])
    # 存储搜索过的坐标
    visited_coordinates = []
    # 存储符合条件的字符
    visited_str = []
    # 记录搜索的最大行数和列数
    row_count, col_count = input_dims[0], input_dims[1]
    for i in range(input_dims[0]):
        for j in range(input_dims[1]):
            find_str(target_str, visited_coordinates, visited_str, [i, j],
                     maps, row_count, col_count)
    # 未找到目标字符串
    print("NO")
def func():
    # please define the python3 input here. 
    # For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    a,b = map(int, input().strip().split())
    dims = [a,b]
    target_str = input().strip()
    input_str = ""
    for i in range(dims[0]):
        input_str += input().strip()+"\n"
    process(dims, target_str, input_str)
if __name__ == "__main__":
    func()
```

```python
# If you need to import additional packages or classes, please import here.
def dfs(W, cnt, i, j, n, m, matrix, is_visited):
    if i >= n or j >= m or i < 0 or j < 0 or is_visited[i][j] == 1:
        return 0
    result = 0
    if matrix[i][j] == W[cnt]:
        if cnt == len(W) - 1:
            return 1
        cnt += 1
        is_visited[i][j] = 1
        result += dfs(W, cnt, i + 1, j, n, m, matrix, is_visited)
        result += dfs(W, cnt, i - 1, j, n, m, matrix, is_visited)
        result += dfs(W, cnt, i, j + 1, n, m, matrix, is_visited)
        result += dfs(W, cnt, i, j - 1, n, m, matrix, is_visited)
        is_visited[i][j] = 0
    else:
        return 0
    return result
def func():
    # please define the python3 input here.
    # For example: a,b = map(int, input().strip().split())
    n, m = map(int, input().strip().split())
    W = input()
    matrix = list()
    for i in range(0, n):
        matrix.append(input())
    is_visited = [[0] * m for i in range(n)]
    # please finish the function body here.
    cnt = 0
    res = 0
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == W[0]:
                for a in range(n):
                    for b in range(m):
                        is_visited[a][b] = 0
                res += dfs(W, cnt, i, j, n, m, matrix, is_visited)
    # please define the python3 output here. For example: print().
    if res > 0:
        print('YES')
    else:
        print('NO')
if __name__ == "__main__":
    func()
```



## **Exponentiation**

**题目：**Problems involving the computation of exact values of very large magnitude and precision are common. For example, the computation of the national debt is a taxing experience for many computer systems.
This problem requires that you write a program to compute the exact value of Rn where R is a real number ( 0.0 < R < 99.999 ) and n is an integer such that 0 < n ≤ 25.

**输入：**The input will consist of a set of pairs of values for R and n. The R value will occupy columns 1 through 6, and the n value will be in columns 8 and 9.

**输出：**The output will consist of one line for each line of input giving the exact value of Rn. Leading zeros should be suppressed in the output. Insignificant trailing zeros must not be printed. Don't print the decimal point if the result is an integer.

**思路：**核心思想是将小数化为整数，并将结果保留为字符串形式，这样就不会漏数。再根据化整过程放大的倍数，将小数点放至合适位置

```python
def func():
    while True:
        try:
            x, y = input().strip().split()
            y = int(y)
            result = ""
            dot_index = x.find(".")
            # int
            if dot_index == -1:
                print(int(x) ** y)
            # float
            else:
                # 去 0
                while x.endswith("0"):
                    x = x[:-1]
                # 变整
                e_i = len(x) - dot_index - 1 # 小数位数
                e_n = 10 ** e_i
                # 去掉小数点，数字扩大10^e_i倍
                x_int = float(x) * e_n  
                x_str = str(int(x_int) ** y)
                e_str = str(e_n ** y)
                # 小数点的位置或补0的数量
                zero_format = len(x_str) - len(e_str) + 1
                if zero_format >=0:
                    result = x_str[:zero_format] + "." + x_str[zero_format:]
                else:
                    result = "." + "0" * abs(zero_format) + x_str
                print(result)
        except:
            break
if __name__ == "__main__":
    func()
```



## **Candy**

**题目：**Solo和koko是两兄弟，妈妈给了他们一大袋糖，每块糖上都有自己的重量。现在他们想要将这些糖分成两堆。分糖的任务当然落到了大哥Solo的身上，然而koko要求必须两个人获得的糖的总重量“相等”（根据Koko的逻辑），要不然就会哭的。
非常不幸的是，koko还非常小，并且他只会先将两个数转成二进制再进行加法，而且总会忘记进位。如当12（1100）加5（101）时：

```
 1100
+0101
——————
 1001
```

此外还有一些例子：

```
5 + 4 = 1
7 + 9 = 14
50 + 10 = 56
# 其实是异或运算:12^5=9,5^4=1
```

现在Solo非常贪心，他想要尽可能使自己得到的糖的**总重量最大，且不让koko哭**。

**输入：**输入的第一行是一个整数N(2 ≤ N ≤ 15)，表示有袋中多少块糖。第二行包含N个用空格分开的整数Ci (1 ≤ Ci ≤ 106)，表示第i块糖的重量

**输出：**如果能让koko不哭，输出Solo所能获得的糖的总重量，否则输出“NO”。

**样例**

```
input:
3
3 5 6
output:
11
解析：三块糖重量为3、5、6，因为5(101)+6(110)=3(11)，所以Solo拿走了重为5和6的糖，koko则得到了重为3的糖

input:
5
1 2 3 4 5
output
NO
解析：五块糖无论如何分，都无法满足koko的要求，所以NO

input:
8
7258 6579 2602 6716 3050 3564 5396 1773
output:
35165
解析：Solo拿走前面7块糖，一共重35165
```

思路：

1. 所有数字进行抑或运算，结果不为0则无法“均分”；
2. 若a^b^c^d= 0,则有任意a^(b^c^d) = 0,即取最轻的那个颗作为a给弟弟即可

```python
'''解释一下异或运算的一些规则:
异或运算具有交换律, 即 a^b = b^a, 与结合律, 即(a^b)^c = a^(b^c);
异或运算的单位元为0, 即 a^0 = 0^a = a, 且 a^a = 0;
异或运算具有自反律, 即 a^b^a = b, 可由规则1和规则2推导出。
根据koko的算法，如果可以“平分”，所有糖果的重量必可分成两份，其中一份的总异或和与另一份的总异或和相等，即

a(1)^a(2)^…^a(k) = a(k+1)^…^a(n)
那么根据规则2，

a(1)^a(2)^…^a(k)^a(k+1)^…^a(n) = 0
即所有糖果重量的总异或和为0，就是重量可以“平分”的条件。
在可以平分的情况下，为使Solo拿到的总重量最大，我们不妨设a(1)为最小的糖果重量，那么将上一个等式左右均与a(1)异或，我们得到

[a(1)^a(2)^…^a(n)]^a(1) = 0^a(1)
根据规则2与规则3，等式化为

a(2)^…^a(n) = a(1)
此时等式左侧即为Solo可以取得的最大重量。'''
from functools import reduce
def func():
    n, l = int(input()), list(map(int, input().split(" ")))
    print("NO" if reduce(lambda x, y: x ^ y, l) else sum(l) - min(l))
if __name__ == "__main__":
    func()
```

```python
def func():
    n = int(input().strip())
    weight = list(map(int, input().strip().split()))
    flag = 0
    for w in weight:
        flag = flag ^ w
    if flag == 0:
        print(sum(weight) - min(weight))
    else:
        print("NO")
if __name__ == "__main__":
    func()
```

```python
print(‘NO’ if reduce(lambda a,b: a ^ b, weight) else (sum(weight) - min(weight)))
```



## 整数拆分

题目：给定一个正整数，我们可以定义出下面的公式:
N=a[1]+a[2]+a[3]+…+a[m];
a[i]>0,1<=m<=N;
对于一个正整数，求解满足上面公式的所有算式组合，如，对于整数 4 :

4 = 4;
4 = 3 + 1;
4 = 2 + 2;
4 = 2 + 1 + 1;
4 = 1 + 1 + 1 + 1;
所以上面的结果是 5 。
注意：对于 “4 = 3 + 1” 和 “4 = 1 + 3” ，这两处算式实际上是同一个组合!

**输入：**每个用例中，会有多行输入，每行输入一个正整数，表示要求解的正整数N(1 ≤ N ≤ 120) 

**输出：**对输入中的每个整数求解答案，并输出一行(回车换行);

```python
'''Python3 简单动态规划及详细解释
这道题是典型的动态规划问题，类似于背包问题和硬币兑换问题。
为了方便理解，我将题目转换成硬币兑换的问题来表述。
假设我们手上有面值为 1，2，3，4，5 等等无数个面值为正整数的硬币，为了凑齐 N 元，总共有多少种硬币组合方法？
我们定义一个数组 DP[a][b]，其值为组合方案的数量，a表示使用多少种硬币，b表示要凑齐的目标值 N 元。
假设 DP[3][9] = 12，表示使用 1，2，3 种硬币，凑齐 9 元一共有 12 种方案。
动态规划的问题，基本上都要找到当前状态和上一个状态的关系表达式。
比如，我们要求 DP[4][10]，这个值和 DP[3][10] 或 DP[4][9] 有什么关系？
我们先来看看 DP[4][10] 和 DP[4][9]。DP[4][10]表示用四种硬币凑齐10元，DP[4][9]表示用
四种硬币凑齐9元，相差 1 元。嗯···，好像不太好想。
换一个思路看看 DP[4][10] 和 DP[3][10]，DP[4][10]表示用 4 种硬币凑齐10元，DP[3][10]表示用 3 种硬币凑齐 10 元，相差 1 种硬币。
DP[4][10] = DP[3][10] + DP[3][10-4] + DP[3][10 - 4*2] = DP[3][10] + DP[3][6]+DP[3][2]+DP[3][0]

什么意思呢？就是说使用 4 种硬币凑齐 10 元，其实就是
1. 使用 3 种硬币，凑齐 10 元
2. 使用 3 种硬币凑齐 6 元，外加一枚 4 元硬币
3. 使用 3 种硬币凑齐 2 元，外加两枚 4 元硬币
4. 只用 4 元硬币，凑齐 10 元
这 4 种情况的方案总数。等号左边和右边的说法是等价的。
这样子我们就可以得到一个表达式：DP[a][b] = DP[a - 1][b] + DP[a - 1][b - a] + DP[a - 1][b - 2a] + ... + DP[a - 1][b - na]
其中 b - na >= 0, n 表示使用 a 元硬币的数量。

DP[4][10] = DP[3][10] + DP[3][6]+DP[3][2]+DP[3][0]
DP[4][6] = DP[3][6] + DP[3][2]+DP[3][0]
所以上面的这个式子还可以再化简为：DP[4][10] = DP[3][10] + DP[4][6], 抽象为DP[a][b] = DP[a - 1][b] + DP[a][b - a]

有关动态规划的题目，我们还需要一个边界条件，容易得到当 a = 1 或 b = 1 时，DP[a][b] = 1。在得到状态转移方程和边界条件后，我们就可以着手编程了。
'''
def divide_int(n):
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(0, n + 1):
            if a == 1:
                dp[a][b] = 1
                continue
            if b == 1:
                dp[a][b] = 1
                continue
            k = b // a
            for i in range(k + 1):
                dp[a][b] += dp[a - 1][b - i * a]
            # dp[a][b] = dp[a - 1][b] + dp[a][b - a]
    print(dp[n][n])
def func():
    while True:
        try:
            a = input()
            divide_int(int(a))
        except EOFError:
            break
if __name__ == "__main__":
    func()
```

```python
# If you need to import additional packages or classes, please import here.
def func():
    # please define the python3 input here. 
    # For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    while 1:
        try:
            N = int(input())
            dp = [[0 for i in range(N+1)] for j in range(N+1)]
            for i in range(1, N+1):
                dp[1][i] = 1
                dp[i][1] = 1
            for i in range(2, N+1):
                for j in range(2, N+1):
                    if j > i:
                        dp[i][j] = dp[i][i]
                    elif j == i:
                        dp[i][j] = 1 + dp[i][j-1]
                    else:
                        dp[i][j] = dp[i-j][j] + dp[i][j-1]
            print(dp[N][N])
            # dp[N][K] N个苹果放K个盘子的不重复方案数
            # if K > N 盘子多了，那dp[N][K] = dp[N][N]
            # if K == N 苹果和盘子一样多，1.没有空盘子，方案数为1
            # 2.至少存在一个空盘子，方案数为dp[N][K-1]，因此dp[N][K] = dp[N][K-1]+1
            # if K < N 苹果多于盘子，1.没有空盘子，则每个盘子至少有一个苹果，那么方案数
            # 跟dp[N-K][K]一致，2.至少一个空盘子，方案数为dp[N][K-1]，因此dp[N][K] = 
            # dp[N][K-1] + dp[N-K][K]
            # 边界：dp[1][N] = 1, dp[N][1] = 1
            """
            5个苹果放3个盘子里，在不空盘的情况下，5 = 3 + 1 + 1；5 = 2 + 2 + 1这个跟
            2 = 2 + 0 + 0；2 = 1 + 1 + 0一致 
            """
        except EOFError:
                break
if __name__ == "__main__":
    func()
```





## 整数拆分 Ver.2

题目：The problem is, given an positive integer N, we define an equation like this:

```
N=a[1]+a[2]+a[3]+…+a[m];
a[i]>0,1<=m<=N;
```

For example, assume N is 5, we can find:

```
5=1+1+1+1+1
5=1+1+1+2
5=1+1+3
5=1+2+2
5=1+4
5=2+3
5=5
```

Note that "5 = 3 + 2" and "5 = 2 + 3" is the same in this problem. Now, you do it!"
But now , you must output all the equations in **lexicographical** order;

**输入：**The input contains several test cases. Each test case contains a positive integer N(1<=N<=20) which is mentioned above. The input is terminated by the end of file.

**输出：**For each test case, you have to output several lines indicate the different equations you have found.

```python
'''DFS'''
def func():
    # please define the python3 input here. 
    # For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    def dfs(cur, s, i, n):
        if s == n:
            print('{}='.format(n),end = '') 
            print('+'.join([str(e) for e in cur]))
            return
        if s > n:
            return
        for j in range(i, n-s+1):
            dfs(cur+[j], s+j, j, n)
    while True:
        try:
            n = int(input())
            dfs([],0,1,n)
        except:
            break
if __name__ == "__main__":
    func()
```

```python
'''1.对于整数n，我们将其题解记为f(n,m) 其中m为等式右边的最小值，初始时 m = 1
2.通过观察我们发现，其等式右边第一个因子总是小于等于n/2的（除 n = n之外）
3.因此 f(n.n) = sum(i + f(n-i,i)) (1 <= i <= n/2)
4.当 n = m时，只能拆分为 n = n，当n = 1时，有 1 = 1'''
def func():
    while True:
        try:
            n = int(input())
            for string in splitEquality(n, 1):
                print(str(n)+"="+string[0:string.__len__()-1])
        except:
            break

def splitEquality(n, m):
    """
        等式拆分递归函数
        :param n 被拆分整数
        :param m 等式右边最小值
        :return 含有等式右边字符串的列表
    """
    resList = list()
    if n == 1:
        resList.append("1+")
        return resList
    if n == m:
        resList.append(str(n)+"+")
        return resList
    x = int(n/2)
    for i in range(m, x+1):
        res = str(i)
        resForI1 = splitEquality(n-i, i)
        for string in resForI1:
            resList.append(res+"+"+string)
    resList.append(str(n)+"+")
    return resList

if __name__ == "__main__":
    func()
```

```c
void dis(int n, int m) {
for(int i = m; i <= n; i++) {
push(i);
if(n - i == 0) {
dump();
} if(n - i >= i) {
dis(n-i, i);
}
pop();
}
}
```

