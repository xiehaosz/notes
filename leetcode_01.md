## 24点游戏

题目：大家都玩过扑克牌(A,2,3…T,J,Q,K)，我们使用T来表示10，且A取值1，J取值11，Q取值12，K取值13，你的任务是判断给定四张牌，能否通过加减乘除四种运算，使得最后的结果是24。若四张牌为A、5、8、J，则可以这么计算5+J+(A*8)=24。

输入：输入四个字符表示四张牌(A,2,3…T,J,Q,K)，用空格隔开。输入到文件末尾结束。

输出：若能计算出24，输出"Yes”，否则输出"No"。

样例：

```
A 5 5 5
A A A A
Yes
No
```

```python
'''https://docs.python.org/zh-cn/3/tutorial/floatingpoint.html#tut-fp-issues'''
from decimal import Decimal
def func(res, num_list):
    if len(num_list) == 1:
        return res == num_list[0]
    for i in range(len(num_list)):
        new_list = num_list.copy()
        new_list.pop(i)
        a = num_list[i]
        if func(res + a, new_list) \
            or func(res * a, new_list) \
            or func(res / a, new_list) \
            or func(a / res, new_list) \
            or func(res - a, new_list) \
            or func(a - res, new_list):
            return True
if __name__ == "__main__":
    while True:
        try:
            swich_list = {'A': 1, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
            num_list = list(map(Decimal, map(lambda i: int(i) if i not in swich_list else swich_list[i], [i for i in input().split(' ')])))
            print('Yes' if func(24, num_list) else 'No')
        except:
            break
```

```python
def func(arr, S):
    # print("arr = %s" % arr)
    # print("S = %s\n" % S)
    # 坑爹的小数精度，不四舍五入算小数的时候会算错
    S = round(S, 2)
    if len(arr) == 1:
        if arr[0] == S:
            return True
        else:
            return False
    for i in range(len(arr)):
        tmp_list = arr[:]
        tmp = tmp_list.pop(i)
        if (func(tmp_list, S-tmp) or func(tmp_list, S+tmp) or func(tmp_list, tmp-S) or func(tmp_list, S/tmp) or func(tmp_list, S*tmp) or func(tmp_list, tmp/S)):
            return True
    return False    
if __name__ == "__main__":
    while True:
        try:
            array = list(input().strip().split())
            for i in range(len(array)):
                if array[i] == "A":
                    array[i] = 1
                elif array[i] == "T":
                    array[i] = 10
                elif array[i] == "J":
                    array[i] = 11
                elif array[i] == "Q":
                    array[i] = 12
                elif array[i] == "K":
                    array[i] = 13
                else:
                    array[i] = int(array[i])
            if func(array, 24):
                print("Yes")
            else:
                print("No")
        except EOFError:
            break
```

## 求对称字符串的最大长度

题目：给定一个字符串（长度<100)，输出该字符串中对称的子字符串的最大长度。比如字符串“hello”，由于该字符串里最长的对称子字符串是“ll”，因此输出2。

输入：google  welcome abccbadef

输出：4 0 6

```python
# 中心扩展法
def func():
    s = input()
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    _max = 0
    for i in range(len(s)):
        _max = max(_max, expand(i, i), expand(i, i + 1))
    print(_max if _max > 1 else 0)
if __name__ == "__main__":
    func()
```

## 最小数字串

题目：给出一个包含n(1<n<100)个元素的正整数数组，将它们连接起来排成一个数，输出能排出的所有数字中最小的一个。例如给定数组{32, 321}，则输出这两个能排成的最小数字32132。注意：不考虑排列后的数字溢出场景。

输入：48,212,32

输出：2123248

```python
# 字符串排序？
if __name__ == "__main__":
    s = sorted(input().strip().split(','))
    num = int(''.join(s))
    print(num)
```

## Who Love Solo Again

**题目描述**

输入一个英文句子，句子中仅包含英文字母，**数字**，空格和标点符号，其中数字、空格和标点符号将句子划分成一个个独立的单词，除去句子中的数字、空格和标点符号，将句子中的每个单词的首字母大写，然后输出句子，输出时各个单词之间以一个空格隔开，**句子以“.”**

输入：只有一行，包含一个长度都不超过100的字符串S，表示英文句子

输出：只有一行，即按要求输出处理后的英文句子，若句子中不含任何单词，则输出一个“.”

```python
#  不用正则
def func():
    line = input()
    for i in range(len(line)):
        if not line[i].isalpha():
            line = line.replace(line[i], ' ', 1)
    line = line.strip().split()
    for i in range(len(line)):
        line[i] = line[i].replace(line[i][0], line[i][0].upper(), 1)
    line = ' '.join(line) + '.'
    print(line)
if __name__ == "__main__":
    func()
```

```python
import re
print(' '.join(list(map(lambda x: x[0].upper() + x[1:], re.sub('[^a-zA-Z]+', ' ', input()).split()))) + '.')
```

```
def func():
    line = ''.join([i if i.isalpha() else ' ' for i in input()])
    line = [i[0].upper() + i[1:] for i in line.strip().split()]
    line = ' '.join(line) + '.'
    print(line)
if __name__ == "__main__":
    func()
```

## 24点游戏

大家都玩过扑克牌(A,2,3…T,J,Q,K)，我们使用T来表示10，且A取值1，J取值11，Q取值12，K取值13，你的任务是判断给定四张牌，能否通过加减乘除四种运算，使得最后的结果是24。
若四张牌为A、5、8、J，则可以这么计算5+J+(A*8)=24。

**输入**：四个字符表示四张牌(A,2,3…T,J,Q,K)，用空格隔开。输入到文件末尾结束

**输出**：若能计算出24，输出"Yes”，否则输出"No"

```python
# 递归可通过（A+B）*（C+D）格式
# If you need to import additional packages or classes, please import here.
def j24(list_input):
    len_input = len(list_input)
    if(len_input == 1):
        return abs(list_input[0] - 24) <=0.001
    for i in range(len_input - 1):
        for j in range(i+1,len_input):
            if j24([list_input[i] + list_input[j]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
            if j24([list_input[i] - list_input[j]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
            if j24([list_input[j] - list_input[i]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
            if j24([list_input[i] * list_input[j]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
            if list_input[j]!=0 and j24([list_input[i] / list_input[j]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
            if list_input[i]!=0 and j24([list_input[j] / list_input[i]] + list_input[0:i] + list_input[i+1:j] + list_input[j+1:]):
                return True
    return False
def func():
    while True:
        try:
            list_tmp = list(input().strip().split())
            list_input = []
            for i in list_tmp:
                if i == "J":
                    list_input.append(11)
                elif i == "Q":
                    list_input.append(12)
                elif i == "T":
                    list_input.append(10)
                elif i == "K":
                    list_input.append(13)
                elif i == "A":
                    list_input.append(1)
                else:
                    list_input.append(int(i))
            if j24(list_input):
                print("Yes")
            else:
                print("No")
        except:
            break
if __name__ == "__main__":
    func()
```

```python
# 暴力
def fun(l1):
    lm=[]
    l2,l3,l4=[],[],[]
    for i in l1:
        l2=l1.copy()
        l2.remove(i)
        for j in l2:
            l3=l2.copy()
            l3.remove(j)
            for k in l3:
                l4=l3.copy()
                l4.remove(k)
                for l in l4:
                    lm.append([i,j,k,l])
    # print(lm)
    lx=[]
    l1 = list(r"+-*/")
    # print(l1)
    for i in l1:
        for j in l1:
            for k in l1:
                mm =[i,j,k]
                lx.append(mm)
                # print(mm)
    for i in lm:
        for j in lx:
            try:
                i=list(map(str,i))
                qq=eval("(("+i[0]+j[0]+i[1]+")"+j[1]+i[2]+")"+j[2]+i[3])
                if qq==24:
                    return "Yes"
                qq=eval("(("+"-1*"+i[0]+j[0]+i[1]+")"+j[1]+i[2]+")"+j[2]+i[3])
                if qq==24:
                    return "Yes"
                qq=eval("("+i[0]+j[0]+i[1]+")"+j[1]+"("+i[2]+j[2]+i[3]+")")
                if qq==24:
                    return "Yes"
                qq=eval("("+"-1*"+i[0]+j[0]+i[1]+")"+j[1]+"("+i[2]+j[2]+i[3]+")")
                if qq==24:
                    return "Yes"
            except:
                pass
    return "No"
while True:
    try:
        a,b,c,d = map(str,input().strip().split())
        l=[]
        for i in a,b,c,d:
            if i =="A":
                l.append(1)
            elif i =="J":
                l.append(11)
            elif i =="Q":
                l.append(12)
            elif i =="K":
                l.append(13)
            else:
                l.append(int(i))
        print(fun(l))
    except:
        break
```

```python
def fun(res, num):
    if len(num) == 1:
        if num[0] == res:
            return True
        else:
            return False
    for i in range(len(num)):
        a = num[i]
        b = num[:]
        b.pop(i)
        if fun(res - a, b) or fun(res * a, b) or fun(res // a, b) or fun(res + a, b) or \
                fun(a - res, b):
            return True
if __name__ == "__main__":
    while True:
        try:
            num = [str(i) for i in input().split()]
            for i in range(4):
                if num[i] == 'A':
                    num[i] = 1
                elif num[i] == 'T':
                    num[i] = 10
                elif num[i] == 'J':
                    num[i] = 11
                elif num[i] == 'Q':
                    num[i] = 12
                elif num[i] == 'K':
                    num[i] = 13
                else:
                    num[i] = int(num[i])
            if fun(24, num):
                print('Yes')
            else:
                print('No')
        except:
            break
```

## 编辑距离

给你两个单词 word1 和 word2（均为小写字母组成），请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

**输入**：第一行为word1，输入第二行为word2

**输出**：两者的编辑距离，即把word1变成word2的最小操作数

```python
def func():
    s1 = input()
    s2 = input()
    s1_len = len(s1)
    s2_len = len(s2)
    # 定义两行dp
    dp = [[i for i in range(s2_len + 1)], [0 if i == 0 else 0 for i in range(s2_len + 1)]]
    for i in range(1, s1_len + 1):
        # 上下两行切换
        temp = i % 2
        dp[temp][0] = dp[temp - 1][0] + 1
        for j in range(1, s2_len + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[temp][j] = dp[temp - 1][j - 1]
            else:
                dp[temp][j] = min(dp[temp - 1][j] + 1, dp[temp][j - 1] + 1, dp[temp - 1][j - 1] + 1)
    print(dp[s1_len % 2][s2_len])
if __name__ == "__main__":
    func()
```

## Treasure Map

Your boss once had got many copies of a treasure map. Unfortunately, all the copies are now broken to many rectangular pieces, and what make it worse, he has lost some of the pieces. Luckily, it is possible to figure out the position of each piece in the original map. Now the boss asks you, the talent programmer, to make a complete treasure map with these pieces. You need to make only one complete map and it is not necessary to use all the pieces. But remember, pieces are not allowed to overlap with each other (See sample 2).

**输入**

The first line of the input contains an integer T (T <= 10), indicating the number of cases.

For each case, the first line contains three integers n m p (1 <= n, m <= 30, 1 <= p <= 100), the width and the height of the map, and the number of pieces. Then p lines follow, each consists of four integers x1 y1 x2 y2 (0 <= x1 < x2 <= n, 0 <= y1 < y2 <= m), where (x1, y1) is the coordinate of the lower-left corner of the rectangular piece, and (x2, y2) is the coordinate of the upper-right corner in the original map.

Cases are separated by one blank line.

**输出**

If you can make a complete map with these pieces, output the least number of pieces you need to achieve this. If it is impossible to make one complete map, just output -1.

```python
class DLX:
    def __init__(self, n, m):
        self.m = m
        self.header = m * n
        self.node_cnt = m * n + 1
        self.lefts = [self.node_cnt - 1] + list(range(self.node_cnt - 1))
        self.rights = list(range(1, self.node_cnt)) + [0]
        self.ups = list(range(self.node_cnt))
        self.downs = list(range(self.node_cnt))
        self.rows = [0] * self.node_cnt
        self.cols = list(range(self.node_cnt))
        self.col_cnt = [0] * self.node_cnt
        self.row_firsts = [0]
        self.min = -1
    def add_row(self, row_index, x1, y1, x2, y2):
        first = self.node_cnt
        for row in range(x1, x2):
            for col in range(y1, y2):
                index = row * self.m + col
                self.rows.append(row_index)
                self.cols.append(index)
                self.col_cnt[index] += 1
                self.ups.append(self.ups[index])
                self.downs.append(index)
                self.downs[self.ups[index]] = self.node_cnt
                self.ups[index] = self.node_cnt
                self.rights.append(first)
                if len(self.lefts) == first:
                    self.lefts.append(first)
                else:
                    self.lefts.append(self.lefts[first])
                self.rights[self.lefts[first]] = self.node_cnt
                self.lefts[first] = self.node_cnt
                self.node_cnt += 1
        self.row_firsts.append(first)
    def _remove(self, c):
        # remove this col from the header
        self.lefts[self.rights[c]] = self.lefts[c]
        self.rights[self.lefts[c]] = self.rights[c]
        i = self.downs[c]
        while i != c:
            # remove this row from the matrix
            j = self.rights[i]
            while j != i:
                self.downs[self.ups[j]] = self.downs[j]
                self.ups[self.downs[j]] = self.ups[j]
                self.col_cnt[self.cols[j]] -= 1
                j = self.rights[j]
            i = self.downs[i]
    def _resume(self, c):
        # reverse the operations done in _remove
        i = self.ups[c]
        while i != c:
            j = self.lefts[i]
            while j != i:
                self.col_cnt[self.cols[j]] += 1
                self.ups[self.downs[j]] = j
                self.downs[self.ups[j]] = j
                j = self.lefts[j]
            i = self.ups[i]
        self.lefts[self.rights[c]] = c
        self.rights[self.lefts[c]] = c
    def get_min_col(self):
        j = self.rights[self.header]
        cur_min = self.col_cnt[j]
        min_col = j
        while j != self.header:
            if self.col_cnt[j] < cur_min:
                cur_min = self.col_cnt[j]
                min_col = j
            j = self.rights[j]
        return min_col
    def dance(self, step):
        if self.min != -1 and self.min <= step:
            return
        if self.lefts[self.header] == self.header:
            if self.min == -1 or step < self.min:
                self.min = step
            return
        c = self.get_min_col()
        if self.col_cnt[c] == 0:
            return
        self._remove(c)
        # remove cols by trying each possible row in this column
        i = self.downs[c]
        while i != c:
            j = self.rights[i]
            while j != i:
                self._remove(self.cols[j])
                j = self.rights[j]
            self.dance(step + 1)
            j = self.lefts[i]
            while j != i:
                self._resume(self.cols[j])
                j = self.lefts[j]
            i = self.downs[i]
        self._resume(c)
def func():
    t = int(input())
    for _ in range(t):
        temp = input().split()
        if not temp:
            temp = input().split()
        n, m, p = map(int, temp)
        d = DLX(n, m)
        pieces = set()
        for i in range(p): 
            piece = input()
            if piece not in pieces:
                pieces.add(piece)
                x1, y1, x2, y2 = map(int, piece.split()) 
                d.add_row(i + 1, x1, y1, x2, y2)
        d.dance(0)
        print(d.min)
if __name__ == "__main__":
    func()
```

```python
# 思路：从0,0开始，找矩形的左上角和右下角分别作为下次递归的起点，寻找合适的碎片， 直到所有矩形的面积等于所有的总面积。

class FindLeastPiece:
    def init(self, nStrs):
    self.length = int(nStrs[0])
    self.width = int(nStrs[1])
    self.size = int(nStrs[2])
    self.allArea = self.length * self.width
    self.result = []

# 获取数据记录，并初始化，剔除重复记录
def GetValidPieces(self):
    self.inputsDict = {}
    for i in range(self.size):
        numbers = [int(item) for item in input().split()]
        numbers.append((numbers[2] - numbers[0]) * (numbers[3] - numbers[1]))
        key = str(numbers[0]) + str(numbers[1]) + str(numbers[2]) + str(numbers[3])
        self.inputsDict[key] = numbers
    tmpDict = {}
    for key in self.inputsDict.keys():
        newKey = str(self.inputsDict[key][0]) + str(self.inputsDict[key][1])
        if newKey in tmpDict:
            tmpDict[newKey].append(self.inputsDict[key])
        else:
            tmpDict[newKey] = [self.inputsDict[key]]
    newList = []
    for key in tmpDict.keys():
        tmpDict[key].sort(key=lambda x: x[4], reverse=True) #关键，可加速处理
        newList.append(tmpDict[key])
    newList.sort(key=lambda x: len(x), reverse=True)
    self.Process(newList)
# 处理记录，排列组合成无重复序号的数列，并进行处理
def Process(self, newList=[], idx=0, inputs=[]):
    if len(inputs) == len(newList):
        self.inputs = dict(zip([str(item[0]) + str(item[1]) for item in inputs], inputs))
        self.gridRecord = 0
        if self.DecideIsValidPiece(self.width, self.length, {}):
            return True
        if self.length != self.width:
            self.gridRecord = 0
            return self.DecideIsValidPiece(self.length, self.width, {})
        return False
    for item in newList[idx]:
        newInputs = inputs[:]
        newInputs.append(item)
        if self.Process(newList, idx + 1, newInputs):
            return True
    return False
#判断两个矩形相交的情况
def IsInsection(self, a, b):
    flag = 1  #相交
    if a[2] <= b[0] or a[0] >= b[2]:
        return 0
    elif a[3] <= b[1] or a[1] >= b[3]:
        return 0
    elif a[0] >= b[0] and a[1] >= b[1] and \
            a[2] <= b[2] and a[3] <= b[3]:
        flag = 2  # b包含a
    elif a[0] <= b[0] and a[1] <= b[1] and \
            a[2] >= b[2] and a[3] >= b[3]:
        flag = 3  # a包含b
    return flag
def SetFlags(self, width, length, item, result={}):
    if item[3] > width or item[2] > length:
        return False  # 超出范围，没找到，继续找
    readyDel = []
    for key in result.keys():
        flag = self.IsInsection(result[key], item)
        if flag == 1:
            return False #与现有状态干涉，没找到，继续找
        elif flag == 2:
            readyDel.append(key)
    iKey = str(item[0]) + str(item[1])
    result[iKey] = item
    self.gridRecord += item[4]
    for key in readyDel:
        self.gridRecord -= result[key][4]
        del result[key]
    return True  #找到并更新了状态
def DecideIsValidPiece(self, width, length, result={}, x=0, y=0):
    if self.gridRecord == self.allArea:
        self.result = result
        return True  # 成功找到，立即结束
    key = str(x) + str(y)
    if key not in self.inputs:
        return False   # 没找到，继续找
    item = self.inputs[key]
    if not self.SetFlags(width, length, item, result):
        return False # 没找到，继续找
    newX, newY = item[2], item[3]
    del self.inputs[key]
    if self.DecideIsValidPiece(width, length, result, newX, y):
        return True
    return self.DecideIsValidPiece(width, length, result, x, newY)
    
if name == “main“:
    number = int(input())
    for count in range(number):
    case = FindLeastPiece(input().split())
    case.GetValidPieces()
    if count < number - 1:
    input()
    if len(case.result) == 0:
    print(“-1”)
    else:
    print(len(case.result))
```

## 鱼缸难题

最近小华买了n条金鱼，小华想买一些鱼缸去装他们，商店有两种鱼缸
第一种：每个鱼缸价格是c1元，可以装n1条鱼
第二种：每个鱼缸价格是c2元，可以装n2条鱼
小华想要把所有的鱼都养在买的那些鱼缸中，而且每个鱼缸都要装满鱼，小华很难计算出两种鱼缸各买多少个最实惠（总花费最少），请你使用程序帮小华计算出最实惠方案。

**输入**

每个用例包含三行
第一行为整数n
第二行为c1,n1
第三行为c2,n2
所有数的范围均为[1,2000000000]

**输出**

每个用例占一行，对于不存在解的情况请输出”failed”（即不能满足所有的鱼都被装在鱼缸中且每个鱼缸都装满）
否则，请输出两个整数m1,m2表示第一种鱼缸买m1个，第二种鱼缸买m2个。保证解是唯一的。

```
这道题涉及鱼缸的价格c和容量n，其实可以转化成单位鱼的价格（c/n）。要想让花费最少,我们要让c/n值最小的方案购买量尽可能的多。

需要判断哪种方案c/n的值最小。
假设方案一的c/n的值最小，我们要尽可能的让方案一的购买量多。方案一的最多购买的数量为：n/n1,此时需要把n-（n/n1）条鱼用方案二装，我们需要保证[n-（n/n1）] % n2 == 0,也就是要保证方案二的鱼缸能装满。然后减少方案一的购买量，使得方案二的鱼缸能装满。方案二最小的时候同上。
```

```python
    # 思路：先求两个鱼缸的存放单条鱼的成本，既然要求总成本最小，其实就是尽可能多的买单价低的鱼缸
然后依次减少单价低的鱼缸的数量，计算剩下的鱼能否正好放在单价贵的鱼缸里面
    single1 = c1 / n1
    single2 = c2 / n2
    # 便宜的鱼缸、贵的鱼缸放的鱼数量
    cheap_n, exp_n = (n1, n2) if single1 < single2 else (n2, n1)
    # 计算出买单价便宜的鱼缸最多能买多少个
    max_cheap_n = n // cheap_n  
    # 依次减少单价便宜的鱼缸的数量，直到小于0
    for i in range(int(max_cheap_n), -2, -1):  
        if i == -1:  # 如果没有解
            print('failed')
            break
        if (n - cheap_n * i) % exp_n == 0:  # 如果剩下的鱼放单价贵的鱼缸里面正好全部放完，输出结果
            cheap_m = i  # 便宜的买多少个
            exp_m = int((n - cheap_n * i) / exp_n)  # 单价贵的买多少个
            if single1 < single2:
                print(cheap_m, exp_m)
            else:
                print(exp_m, cheap_m)
            break
```

```python
judge_avg=(c1/n1) < (c2/n2)
m1=-5
m2=-5
if judge_avg:
    for i in range(n//n2+1):
        if (n-i*n2)%n1==0:
            m1=(n-i*n2)//n1
            m2=i
            break
else:
    for i in range(n//n1+1):
        if (n-i*n1)%n2==0:
            m2=(n-i*n1)//n2
            m1=i
            break
if m1==-5:
    print('failed')
else:
    print('{} {}'.format(m1,m2))
if name == “main“:
func()
```

## 同网CS

题目：判断IP是否同一网段

IP规则如下：

1. 网络标识不能以数值127开头(以127开头的地址为特殊地址，比如127.0.0.1是loopback IP)
2. 网络标识第一个字节不能是255和0C.
3. IP每个字段不能大于255

掩码规则如下：

1. 不能全部是255
2. 不能全部是0
3. 掩码的高位(bit)必须是连续的1，如这种掩码255.255.253.0 —> 11111111.11111111.111111**01**.00000000是错误的

**输入**

第一行是我的电脑的IP地址
第二行是我的电脑的子网掩码
第三行整数N，表示后面N个同学的IP地址
第1个同学的IP地址……

第N个同学的IP地址

**输出**

计算并输出N个IP地址是否与我的电脑在同一子网内。
对于在同一子网的输出：let's rock
对于在不同子网的输出：not you
对于无效的联网IP输出：Invalid IP address.
对于无效的子网掩码：Invalid netmask address.

```python
def check_ip(ip):
    ip = list(map(int, ip.strip().split('.')))
    if ip[0] in [127, 255, 0] or any([i > 255 for i in ip]):
        return False, 'Invalid IP address.'
    else:
        return True, ip

def check_mask(mask):
    mask = list(map(int, mask.strip().split('.')))
    mask_b = ''.join(map(bin, mask)).replace('0b', '')
    if '01' in mask_b or sum(mask) in [0, 1020]:
        return False, 'Invalid netmask address.'
    else:
        return True, mask


def mask_ip(ip, mask):
    return [i & m for i, m in zip(ip, mask)]


def func():
    ok, ip = check_ip(input())
    if ok:
        ok, mask = check_mask(input())
        if ok:
            my_net = mask_ip(ip, mask)
        else:
            print(mask)
            return
    else:
        print(ip)
        return
    n = int(input())
    for i in range(n):
        ok, ip = check_ip(input())
        if ok:
            if my_net == mask_ip(ip, mask):
                print('let\'s rock')
            else:
                print('not you')
        else:
            print(ip)

if __name__ == "__main__":
    func()
```

## 数对数（未解决）

**题目描述**

有n个人站成一排，每个人都有一个数字，令第i个人的数字是Ai。如果Ai和Aj(i < j)两个数字有相同的素因子，而且对于任意的Ak(i<k<j)都不含有该因子，那么这两个人就是一对朋友。
现在对于给定的l和r(1<=l<r<=n)，我们把这行分成3个部分[1,l],[l+1,r-1],[r,n]，然后我们把[1,l]∪[r,n]视为区域A，[l+1,r-1]视为区域B（如果l=r-1，那么B是一个空区域），然后我们定义F(l,r)为区域A中的朋友对数减去区域B中的朋友对数的值。现在，我们有一个最喜爱的值C，你能够找出所有满足F(l,r)=C的数对吗？

**输入**：第一行输入两个整数n和C(1<=n<=5000, |C|<n*n)。第二行输入n个数A1, A2, A3, …, An，表示这n个人的数字，对于任意的1<=i<=n，1<=Pi<=10000000。

**输出**：一行一个整数，表示满足F(l,r)=C的对数。

思路：Couples简单题解

首先可以用欧拉打表法打出1~10000000以内的素数，并且找出非1的整数的一个质因子。
然后就是构图，对于序列进行分析构图，确定所有的朋友关系。
然后枚举r和l，随着l的偏移，一些答案会跟着变化，里面转移的复杂度是O(1)，整体复杂度O(n*n)。

样例

```
输入：9 2
输入：1 3 6 7 15 4 12 1 14
输出：4
```

## 难弄的排列组合

**题目：**一个排列组合问题：有m个盒子（盒子与盒子之间是不同的），n个球(球与球之间是完全相同的)，要求每个盒子里至少放k个球，共有多少种不同的方法？

**输入：**每组测试用例占一行，包含三个整数m,n,k(1<=m<=100,1<=n<=1000,0<=k<=20)

**输出：**共有多少种不同的方法，由于结果可能超出整数int范围，输出对5201314取余后的值。

提示：首先，我们把必须要放的k个去掉，即n=n-m*k，接着，问题转换为n个小球，放在m个盒子里共有多少种方法，是经典的排列组合问题。答案是(n+m-1)C(m-1)。
求组合数使用公式(n+1)Cm=nCm+nC(m-1)。

```python
m, n, k = map(int, input().strip().split())
def A(num: int) -> int:
    """
求出num的阶乘
    :rtype: int
    """
    re = 1
    for i in range(1, num + 1):
        re *= i
    return re
n = n - m * (k - 1)  # 每个盒子至少放k个球，先每个盒子放入(k-1)个球，再使用挡板法
result = A(n - 1) // A(m - 1) // A(n - m) % 5201314  # 用/会超出float最大值，所以用//，这样返回的是integer
print(result)
```

```python
def func():
    m, n, k = map(int, input().strip().split())
    n = n - m * k
    if n < 0:
        print(0)
        return
    c, c1, c2 = 1, 1, 1
    for i in range(1, n+m):
        c *= i
    for j in range(1, m):
        c1 *= j
    for k in range(1, n + 1):
        c2 *= k
    print(c // c1 // c2 % 5201314)
if __name__ == "__main__":
    func()
```

```python
# If you need to import additional packages or classes, please import here.
from math import factorial
def func():
    # please define the python3 input here. 
    # For example: a,b = map(int, input().strip().split())
    # please finish the function body here.
    # please define the python3 output here. For example: print().
    m, n, k = map(int, input().strip().split())
    if m*k>n:
        print(0)
        return
    down = n-m*(k-1)-1
    up = m-1
    print((factorial(down)//factorial(up)//factorial(down-up))%5201314)
if __name__ == "__main__":
    func()
```



## 牛郎数星星

每年的七夕，牛郎和织女在鹊桥相会，遥望着漫天星河，发现刚好有若干颗星星从左到右排成一排，形成了一条星链。细心的牛郎依次给这些星星标上了序号。一共有n颗星星，序号从最左边的1开始，从左到右依次递增，直到最右边的n号星星。调皮的织女给这些星星分别设定了星愿值。织女想知道从第x号星星到第y号星星之间（包括x和y）有多少种不同的星愿值，你能帮助牛郎尽量快的回答织女的问题吗？

**输入**

第一行输入两个正整数n, q(1 <= n, q <= 100000)，表示有n颗星星和织女的q次操作询问。
第二行输入n个正整数a_1, a_2, a_3, …, a_n(对于任意的1 <= i <= n，都有1 <= a_i <= n), 依次表示第1号星星到第n号星星的初始星愿值。
每次织女问一个问题，牛郎就要回答出来，这里为了必须要求在线，所以我们记录一个pre，表示上一次询问的答案，每组测试数据一开始pre = 0。
接下来q行，每行包含2个正整数x’, y’，真实的查询区间是[x’ - pre, y’ - pre * 2]，数据的输入保证1 <= x’ - pre <= y’ - pre * 2 <= n，查询出这个区间里有多少个不同的星愿值。

**输出**

对于每个询问，输出一行一个整数，表示询问的答案，并把pre置为当前的答案。

**提示**  https://zhuanlan.zhihu.com/p/250565583

题目大意：
给定 n 个数字，要求在线查询区间[l, r]所形成的集合的元素个数。
题解：
我们先考虑离线的做法：
假设可以离线，那么我们可以用树状数组来实现。区间的排序按照 r 的从小
到大排序，然后按照双指针的走法，按照数组从左到右依次加入每个值，加入值
的时候按照这样的准则：如果前面出现过这个值，那么把最近出现这个值的位置
-1，把当前位置+1，如果没有出现过，那么直接对当前位置+1。这样处理之后，
那么对于每个区间，当走到当前的右端点时，那么直接查询树状数组中[l, r]
的和，那么就做好了。
现在要求在线，其实算法思想和上面是一样的，利用可持久化线段树来维护，
每次最多修改 2 个节点，建树在输入结束后就建立好，那么每次对[l, r]进行
查询只要查询第 r 颗线段树内[l, r]的区间和即可。

```python
# 可持久化线段树
# 节点类
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 建空数
def build_tree(l,r):
    node = Node(0)
    if l==r:
        return node
    else:
        mid = (l + r)//2
        node.left = build_tree(l, mid)
        node.right = build_tree(mid+1, r)
    return node
# 插入叶子
def insert(index, val, fatherNode, rangeL, rangeR, val2):
    if rangeL == rangeR:
        return Node(val2)
    else:
        node = Node(fatherNode.val, fatherNode.left, fatherNode.right)
        mid = (rangeL + rangeR)//2
        if mid >= index:
            nodel = insert(index, val, node.left, rangeL, mid,val2)
            node.left = nodel
            node.val += val
        else:
            nodeR = insert(index, val, node.right, mid+1, rangeR,val2)
            node.right = nodeR
            node.val += val
    return node
# 查询
def query(node, queryL, queryR, rangeL, rangeR, rs=0):
    if queryL <= rangeL and queryR >= rangeR:
        newrs = rs + node.val
    else:
        mid = (rangeL + rangeR)//2
        newrs = 0
        if queryL <= mid:
            temprs = query(node.left, queryL, queryR, rangeL, mid, rs)
            newrs = rs + temprs
        if queryR > mid:
            temprs = query(node.right, queryL, queryR, mid+1, rangeR, rs)
            newrs = newrs + temprs
    return newrs
def func():
    n, q = map(int,input().split(" "))
    list_ = input().split(" ")
    hash_index = {}
    root = build_tree(1, n)
    root_list = [0]*(n+1)
    root_list[0] = root
    for i in range(1,n+1):
        i_1 = i-1
        temp_root = insert(i, 1, root_list[i_1], 1, n,1)
        last_index = hash_index.get(list_[i_1])
        if last_index:
            temp_root = insert(last_index, -1, temp_root, 1, n,0)
        root_list[i] = temp_root
        hash_index[list_[i_1]] = i
    del list_,root,hash_index
    pre = 0
    for _ in range(q):
        ql, qr = map(int,input().split(" "))
        ql -= pre
        qr -= pre*2
        node = root_list[qr]
        rs = query(node, ql, qr, 1, n, rs=0)
        print(rs)
        pre = rs
func()
```



## 所有的平方差  #

**题目描述**

我们发现有一些数具备这样的性质：可以表示成两个自然数的平方之差。比如3这个数字，可以表示成3=2*2-1*1；有时这种表示方法不止一种，例如9，可以有这两种表示：9=3*3-0*0和9=5*5-4*4；但是也可能出现不存在的情况，例如6没有办法表示成任何的两个自然数平方差。

现在我们要求找出给定的n所有的平方差，例如有m对符合要求，那么假设这m对表示成:

n = a_1*a_1 - b_1*b_1, n = a_2*a_2 - b_2*b_2, … , n = a_m*a_m - b_m*b_m

其中对于任意的1<=k1,k2<=m，a_k1 != a_k2 或者 b_k1 != b_k2，并且a_k1,a_k2,b_k1,b_k2>=0,均为自然数.

那么我们需要求的是n^(a_1*a_1+b_1*b_1+a_2*a_2+b_2*b_2+…+a_m*a_m+b_m*b_m) % 大质数10000000019的值

**解答要求**时间限制：1000ms, 内存限制：100MB

**输入**

输入数据只有一行一个整数n(1<=n<=2^62)

**输出**

输出题目中的答案，若不存在，则输出-1。

**样例**

输入样例 1 复制

```
3
```

输出样例 1

```
243
```

**提示**

根据平方差公式，对n=a*a-b*b可以因式分解成n=(a+b)*(a-b)，于是不难发现a+b和a-b都是n的两个因子，对于每组不同的满足这种关系的因子对肯定是不一样的答案，于是我们就要枚举出n的所有的因子出来，然后不难发现，对于n=x*y(x>=y)，要满足a+b=x并且a-b=y的情况，必须x和y的奇偶性相同，那么如果说n是2的倍数但不是4的倍数的话，那么不可能找到任何的因子对了，如果n是奇数，那么所有满足条件的因子对都是答案，如果n是偶数，那么必然两个因子里面都要含有2这个素因子，于是我们可以得到所有的因子对，这里大整数分解用**Pollard rho算法**，复杂度约为O(n^0.25)。
接下来就是算n^x % mod了，这里x很大，可能会超long long，如果n%mod=0，显然答案是0了，否则根据mod是大质数的原则，于是方法是用费尔马小定理，n^x % mod = n^(x % (mod - 1) + mod) % mod，根据同余定理，指数都在long long范围内，然后计算n^y % mod，然后用快速幂，在log(y)时间内可以计算出来，当然mod * mod是超过long long的，中间需要用防溢出的快速乘法。

```python
from random import randint
Big_Prime_Num = 10000000019
def quickMulMod(a,b,m):
    '''a*b%m,  quick'''
    ret = 0
    while b:
        if b&1:
            ret = (a+ret)%m
        b//=2
        a = (a+a)%m
    return ret
def quickPowMod(a,b,m):
    '''a^b %m, quick,  O(logn)'''
    ret =1
    while b:
        if b&1:
            ret = (ret*a) % m
        b >>= 1
        a = (a*a) % m
    return ret
def isPrime(n,t=5):
    '''miller rabin primality test,  a probability result
        t is the number of iteration(witness)
    '''
    t = min(n-3,t)
    if n<2:
        print('[Error]: {} can\'t be classed with prime or composite'.format(n))
        return
    if n==2: return True
    d = n-1
    r = 0
    while d%2==0:
        r+=1
        d//=2
    tested=set()
    for i in range(t):
        a = randint(2,n-2)
        while a in tested:
            a = randint(2,n-2)
        tested.add(a)
        x= quickPowMod(a,d,n)
        if x==1 or x==n-1: continue  #success,
        for j in range(r-1):
            x= quickMulMod(x,x,n)
            if x==n-1:break
        else:
            return False
    return True
# 求最大公约数
def gcd(a,b):
    while b!=0:
        a,b=b,a%b
    return a
# 求质因数
def factor(n):
    '''pollard's rho algorithm'''
    if n==1: return []
    if isPrime(n):return [n]
    fact=1
    cycle_size=2
    x = x_fixed = 2
    c = randint(1,n)
    while fact==1:
        for i in range(cycle_size):
            if fact>1:break
            x=(x*x+c)%n
            if x==x_fixed:
                c = randint(1,n)
                continue
            fact = gcd(x-x_fixed,n)
        cycle_size *=2
        x_fixed = x
    return factor(fact)+factor(n//fact)
def func():
    pow_sum = 0
    if num == 1:
        return print(1)
    prim_factor_lst = factor(num)    # 质因数列表，例如 [x1, x1, x2, x2, x3, x4]
    prim_factor_set = set(prim_factor_lst)    # 质因数去重，例如 {x1, x2, x3, x4}
    # 质因数临时数组，例如 [[1, x1, x1^2], [1, x2, x2^2], [1, x3], [1, x4]]
    tmp_factor_lst = [[x**i for i in range(prim_factor_lst.count(x) + 1)] for x in prim_factor_set]    
    all_factor_lst = tmp_factor_lst[0]
    for lst in tmp_factor_lst[1:]:
        # 所有因数
        all_factor_lst = [i*j for i in all_factor_lst for j in lst] 
    all_factor_lst.sort()    # 排序后，从左往右取x，从右往左取y
    count_all_factor = len(all_factor_lst)
    # 当所有的因数为奇数个，此时中间的那个因数既为x又为y，在计算因数对数时，需要另外考虑
    if count_all_factor % 2 == 1:
        count_loop = count_all_factor//2+1    # 因数对数
    else:
        count_loop = count_all_factor//2
    for i in range(count_loop):
        # 从左往右取x，从右往左取y
        x, y = all_factor_lst[i], all_factor_lst[count_all_factor - i - 1]
        if (x % 2 == 0 and y % 2 == 0) or \
            (x % 2 == 1 and y % 2 == 1):    # 因数对的奇偶性相同才符合要求
                a, b = (x + y) // 2, (x - y) // 2
                pow_sum += a**2 + b**2
    if pow_sum == 0:
        res = -1
    else:
        # 费马小定理、快速幂求模
        res = quickPowMod(num, pow_sum % (Big_Prime_Num - 1), Big_Prime_Num) 
    print(res)
if __name__ == "__main__":
    num = int(input().strip())
    func()
```

