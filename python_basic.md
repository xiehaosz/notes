https://github.com/junxiaosong/AlphaZero_Gomoku



https://zhuanlan.zhihu.com/p/105467597

光学模拟

https://ricktu288.github.io/ray-optics/simulator/?zh-CN

# 算法复杂度

```python
# 通过求x的n次方理解时间复杂度
x = 6
n = 10
repeat = 1000

# 最直观的方法是for循环, 循环次数为n, 时间复杂度O(n)
def for_loop(x, n):
    res = 1
    for i in range(n):
        res *= x
    return res
t0 = time.time()
for i in range(repeat):
    res = for_loop(x, n)
t1 = time.time()
print(res, t1-t0)

# 递归, 时间复杂度O(n), 每一次递归n-1直到n=0, 调用n次，时间复杂度也是O(n)
def func_recur(x, n):
    if n == 0:
        return 1
    return func_recur(x, n-1) * x
t0 = time.time()
for i in range(repeat):
    res = func_recur(x, n)
t1 = time.time()
print(res, t1-t0)

# 优化递归, 满二叉树, 时间复杂度O(n)
def func_recur_2(x, n):
    if n == 0:
        return 1
    if n % 2 == 1:
        # 这个地方有两次递归调用
        return func_recur_2(x, int(n/2)) * func_recur_2(x, int(n/2)) * x
    return func_recur_2(x, n/2) * func_recur_2(x, n/2)
t0 = time.time()
for i in range(repeat):
    res = func_recur_2(x, n)
t1 = time.time()
print(res, t1-t0)

# 优化在上一基础上优化递归, 时间复杂度O(logn)
def func_recur_3(x, n):
    if n == 0:
        return 1
    t = func_recur_3(x, int(n/2))
    if n % 2 == 1:
        return t * t * x
    return t * t
t0 = time.time()
for i in range(repeat):
    res = func_recur_3(x, n)
t1 = time.time()
print(res, t1-t0)
```



# 二分查找（binary search）

https://oi-wiki.org/basic/binary/

```python
# 递归实现
def binarySearch (arr, l, r, x): 
    """
    :param arr: 升序排列数组
    :param l: 检索的起始元素索引
    :param r: 检索的数组长度
    :param x: 检索值
    :return:  检索值对应的索引
    """
    # 基本判断
    if r >= l: 
        mid = int(l + (r - l)/2)
        # 元素位于中间位置
        if arr[mid] == x: 
            return mid 
        # 元素小于中间位置的元素，继续比较左边的元素
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        # 元素大于中间位置的元素，继续比较右边的元素
        else: 
            return binarySearch(arr, mid+1, r, x) 
    else: 
        # 不存在
        return -1
  
# 测试数组
arr = [ 2, 3, 4, 10, 40, 50, 80, 100 ] 
x = 10
  
# 函数调用
result = binarySearch(arr, 0, len(arr)-1, x) 
if result != -1: 
    print ("元素在数组中的索引为 %d" % result )
else: 
    print ("元素不在数组中")
```



```python
# 顺序查找, 时间复杂度O(n)
def func_1(num, target):
    i, l = 0, len(num)
    while i < l:
        if num[i] == target:
            return i
        i += 1
    return -1
print(func_1(num, target))

# 二分查找, 每次查找的范围缩小一半, 时间复杂度O(logn)
def func_2(num, target):
    left, right = 0, len(num)
    while left < right:
        middle = left + (right-left) >> 1
        if num[middle] == target:
            return middle
        elif target > middle:
            left = middle + 1
        else:
            right = middle
    return -1
print(func_2(num, target))
```



# 贪心

步骤1：从某个初始解出发；
步骤2：采用迭代的过程，当可以向目标前进一步时，就根据局部最优策略，得到一部分解，缩小问题规模；
步骤3：将所有解综合起来。

缺点：每一步的最优并不一定是全局最优

例如使用个数最少的不同面额硬币组成期望值，贪心原则就是尽可能使用大面额硬币，个数不一定是最少的。

## 最大子序和（贪心）

```
时间:O(n),一次遍历; 空间:O(1), 常数 
记录[当前值, 之前和, 当前和, 最大和]， 指针不断前移
最大和 = 当前和 if 当前和 > 最大和
若之前和小于0, 则丢弃之前的数列从当前开始累积
```



# 动态规划（Dynamic Planning）

https://www.luogu.com.cn/blog/pks-LOVING/junior-dynamic-programming-dong-tai-gui-hua-chu-bu-ge-zhong-zi-xu-lie

关于动态规划的典型题目1：有一个长度为n的数列，找出其中最长的上升序列长度

嵌套循环遍历，复杂度O(n^2)，每一个元素，都顺序向后查找直到出现降序，得的每一个元素对应的最长升序长度

蠕虫法(我起的名字)，复杂度O(nlogn)，从第一个元素开始，找到最长升序尾部为j，下一次从j+1开始，后续的每一次查找都从上一次的尾部+1开始，像虫子一样伸缩前进

关于动态规划的典型题目2：获得两个序列中的最长公共子序列LCS（least common sequence）

## 最大子序和（动规）

```python
前一个元素大于0, 则加到当前元素上（更新）
取最大元素
动态规划转移方程: f(i) = max{f(i-1)+num[i], num[i]}
```



# 相似度计算

https://yishuihancheng.blog.csdn.net/article/details/89927608?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.highlightwordscore&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.highlightwordscore

https://zhuanlan.zhihu.com/p/342861735

# 前缀/中缀/后缀表达式

https://blog.csdn.net/Antineutrino/article/details/6763722

https://blog.csdn.net/weixin_44520259/article/details/103721064

https://python.iitter.com/other/260684.html

```python
"""
文本表达式数据分析
Created on 2022-1-30
"""

import re
import numpy as np
import pandas as pd

# 运算符优先级字典 Mathematical Operator Priority
OPTR_PRI = {'^': 3,  # 指数
            '%': 2,  # 取余
            '*': 2,  # 乘法
            '/': 2,  # 除法
            '+': 1,  # 加法
            '-': 1,  # 减法
            '(': 0,
            ')': 0
            }

# # 条件运算符优先级字典 Conditional Operator Priority
COND_PRI = {'=': 2,   # 等于
            '>': 2,   # 大于
            '<': 2,   # 小于
            '>=': 2,  # 大于等于
            '<=': 2,  # 小于等于
            '!=': 2,  # 不等于
            'in': 2,  # 属于
            '&': 1,   # 与
            '|': 1,   # 或
            '(': 0,
            ')': 0
            }

# 预定义操作函数 Operator Functions,代码中的函数关键字定义必须为大写字母,允许嵌套函数或计算表达式,例如: AVG(10*LOG([x])+12)/1000
OPRA_FUNCS = ['AVG',
              'MAX',
              'MIN',
              'SUM',
              'INT',   # 取整数
              'RND',   # round, 四舍五入
              'LOG',   # log10, 以10为底的对数
              'CNT',   # count, 计数
              'MODE',  # mode, 众数
              'UNI',   # unique, 返回不重复值列表,使用时必须作为最外层函数
              ]

# 预定义专用函数 Specific Functions,代码中的函数关键字定义必须为大写字母,带额外参数的函数,使用','分隔参数
SPEC_FUNCS = ['PERCENT',  # PERCENT(series, 可选x), 返回每个不重复元素的百分比, 指定x参数则返回指定元素x的百分比
              'CNTIF',    # COUNTIF(series, 可选condition), 无条件参数返回分组计数结果, 有条件返回条件计数结果
              'LARGE',    # LARGE(series, n, 可选False), 返回第n大值, 最后一个参数为True返回前n大值列表
              'SMALL',    # SMALL(series, n, 可选False), 返回第n小值, 最后一个参数为True返回前n小值列表
              'FILTER',   # FILTER(series, condition), 返回条件筛选后的数据系列
              'GROUPBY'   #
             ]


class Stack(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()


class ExpnResolution:
    def __init__(self, expn):
        """
        :param expn: 计算表达式（中缀表达式)
        表达式规则: 表达式为数值,有效变量,有效运算符,以及有效函数的合法组合
        1. 变量必须使用[]包围
        2. 有效运算符有内置字典MO_PRI定义
        3. 有效函数名由内置列表OPRA_FUNCS定义,函数内容由()包围
        4. 如存在筛选条件,将条件表达式放入{}中
        5. 条件表达式中匹配的文本内容需要使用''或""包围
        6. 如条件表达式内嵌套计算表达式, 使用':'标识起始位置, 注意嵌套计算表达式会被其返回字符串替换, 请确保替换后条件表达式仍然有效

        示例1： 计算表达式 '10*LOG([xx跟踪/A列]*2 + [xx跟踪/B列]*(3+1)) * SUM([xx跟踪/C列])'
        示例2： 有条件的计算表达式 '10*LOG([xx跟踪/A列]*2  {[xx跟踪/B列]=='Pass'}' 注意文本使用''
        示例3： 有嵌套条件的计算表达式 '([xx跟踪/A列]/1000) {[xx跟踪/C列] in :UNI([yy跟踪/A列]  {[yy跟踪/B列]==1}}'
        """
        self.expn = expn

        # 表达式中的变量数据组成DataFrame
        self.df_dict = dict()
        self._read_vars(expn.replace(' ', ''))

        # 条件表达式匹配
        pat_cond = re.compile(r'(.*?)\{(.*)\}')
        cond_search = pat_cond.search(expn)
        if cond_search:
            # 存在筛选条件
            expn, cond = cond_search.groups()
            expn = self._check_expn(expn)

            if ':' in cond:
                # 存在嵌套计算表达式
                pat_sub_expn = re.compile(r':(.*)')
                sub_expn = pat_sub_expn.search(cond).group(1)
                # 转换为嵌套表达式的返回结果(字符串)
                sub_res = ExpnResolution(sub_expn).result
                cond = pat_sub_expn.sub(str(sub_res), cond)
            try:
                # 无嵌套的最小条件表达式,应用条件筛选
                cond, trace = self._check_cond(cond)
                if trace is not None:
                    self.df_dict[trace].query(cond, inplace=True)
            except Exception as e:
                if not isinstance(e, Exception):
                    pass
                    # TODO 条件表达式错误
        else:
            expn = self._check_expn(expn)

        # 保存表达式的计算结果
        self.result = None
        post_stack = self._infix_to_postfix(expn)
        if post_stack is not None:
            self.result = self._calc_postfix(post_stack)

    @ staticmethod
    def _check_expn(expn):
        """
        :param expn: 字符串表达式
        :return: 返回合法的字符串表达式, 如果表达式不合法返回None
        """
        # 清除空格和起始标识':'
        expn = expn.replace(' ', '')
        # 表达式局部为有效负数时, 插入一个数字0方便处理
        if expn[0] == '-':
            expn = '0'+expn
        expn = expn.replace('(-', '(0-')

        # 兼容处理
        expn = re.sub(r'([\+|\-|\*|\/|\%|\^|\(|\)])(\.\d+)', r'\g<1>0\g<2>', expn)
        expn = re.sub(r'(\d+\.)([\+|\-|\*|\/|\%|\^|\(|\)])', r'\g<1>0\g<2>', expn)
        expn = expn.replace('+-', '-')
        expn = expn.replace('-+', '-')
        expn = expn.replace('--', '+')
        expn = expn.replace('**', '^')
        expn = expn.replace('×', '*')
        expn = expn.replace('(+', '(')
        expn = expn.replace('()', '')

        # 异常表达式规则
        if expn.count('[') != expn.count(']'):                      # 缺失中括号
            return None
        if expn.count('(') != expn.count(')'):                      # 缺失小括号
            return None
        if expn.count('{') != expn.count('}'):                      # 缺大括号
            return None
        pat_err_rules = list()
        pat_err_rules.append(r'[\+\-\*\/\%\^][\+\-\*\/\%\^\)]')     # 连续运算符
        pat_err_rules.append(r'\([\*\/\%\^]')                       # 左括号接运算符
        pat_err_rules.append(r'\.\d*\.')                            # 连续点号数字
        # 可增加规则

        for pat in pat_err_rules:
            if re.search(pat, expn):
                raise ValueError('输入表达式存在错误, 请核查.')
        return expn

    @ staticmethod
    def _check_cond(cond):
        """
        :param cond: 最小条件表达式（无嵌套）
        :return: 表达式的数据源, 兼容处理后的表达式
        """
        # 清除空格
        cond = cond.strip(' ')
        # 兼容处理
        cond = cond.replace('>>', '>')
        cond = cond.replace('<<', '<')
        cond = cond.replace('>>', '>')
        cond = cond.replace('||', '|')
        cond = cond.replace('=<', '<=')
        cond = cond.replace('=>', '>=')
        cond = cond.replace('<>', '!=')
        cond = cond.replace('&&', '&')
        cond = re.sub(r'or', '|', cond, flags=re.I)
        cond = re.sub(r'and', '&', cond, flags=re.I)
        cond = re.sub(r'([^>|<|=])(=)([^=])', r'\1==\3', cond)
        cond = re.sub(r'^=([^=])', r'==\1', cond)

        vars = set(re.findall(r'\[.*?\]', cond))
        if vars:
            vars_pack = [var.strip('[]').split('/') for var in vars]
            trace_list, key_list = zip(*vars_pack)
            if len(set(trace_list)) > 1:
                # 判断最小条件表达式不能跨数据源
                return False
            # 变量名转换为列标题
            cond = cond.replace('['+trace_list[0]+'/', '')
            cond = cond.replace(']', '')
            return cond, trace_list[0]
        else:
            return cond, None

    @ staticmethod
    def _check_brief_cond(var, cond):
        """
        :param var: 变量名称
        :param cond: 条件表达式
        :return: 兼容处理类似 '>2'这种简略表达式, 返回完整形式 'var>2'
        """
        cond, _ = ExpnResolution._check_cond(cond)
        if var not in cond:
            # 如果条件表达式中没有变量名, 为每个条件运算符前增加变量名
            cond = re.sub('([><=!]+)', r'%s\1' % var, cond)
        return cond

    def _read_vars(self, expn):
        """
        :return: 解析表达式中的变量并读取对应数据, 转化为Pandas Series存入字典
        """
        for var in set(re.findall(r'\[.*?\]', expn)):
            # 暂定义外部数据源的命名规则使用'/'分隔符[跟踪名/数值标题]
            if '/' in var:  # 有效的变量信息
                trace, key = var.strip('[]').split('/')

                # TODO 根据信息读取数据,同源数据组帧.如果有数据要合并或对齐的需求另写函数

                # 这里的trace使用代码内置样本数据kpi1和kpi2进行调试
                if trace not in self.df_dict:
                    self.df_dict.setdefault(trace, pd.DataFrame())
                if key not in self.df_dict[trace]:
                    self.df_dict[trace][key] = eval(trace)[key]

    def _get_vars(self, var):
        """
        :param var: 由'[]'包围的变量名称
        :return: 根据关键字从self.df_dict中取数据
        """
        trace, key = var.strip('[]').split('/')
        return self.df_dict[trace][key]

    @ staticmethod
    def _check_var_name(var_name: str) -> str:
        # 变量名作为列标题时如果存在'.','/','['等各种符号时会出现很多异常, 统一替换为'_'后返回
        return var_name.strip('[]').replace('/', '_')

    @ staticmethod
    def _search_func_content(func_name, expn):
        """
        :param func_name: 函数名称
        :param expn: 表达式
        :return: 从表达式中提取出指定函数的内容部分
        """
        pat_foo = re.compile(r'%s\(' % func_name)
        end = len(expn)
        foo = pat_foo.search(expn)
        if foo:
            brac_pair = 1
            idx0, idx1 = foo.span()
            while brac_pair > 0 and idx1 < end:
                if expn[idx1] == '(':
                    brac_pair += 1
                elif expn[idx1] == ')':
                    brac_pair -= 1
                idx1 += 1
            if brac_pair > 0:
                return False
            else:
                return expn[idx0:idx1]

    @ staticmethod
    def _infix_to_postfix(expn) -> (Stack, bool):
        """
        :param expn: 中缀表达式
        :return: 将中缀表达式转化为后缀表达式栈
        算法流程:
        1.初始化两个栈：运算符栈S1和结果栈S2（S1的出栈元素压入S2）
        2.遇到操作数, 压入S2
        2.遇到函数, 压入S1
        3.遇到运算符, 如果S1为空或栈顶运算符为左括号'(', 压入S1
        4.否则, 弹出所有优先级大于或等于该运算符优先级的S1栈顶运算符
        5.遇到右括号')'：S1持续出栈,遇到'('停止,丢弃左括号
        6.如果'('属于函数,弹出函数
        7.结果栈输出
        """
        # 清除空格
        expn = expn.replace(' ', '')
        # 初始化临时操作栈, 输出结果栈
        s1, s2 = Stack(), Stack()

        # 匹配变量操作数
        pat_operand = re.compile(r'\[.*?]', re.I)
        # 匹配数值操作数
        pat_digital = re.compile(r'\d+\.?\d*')
        # 匹配预定义专用函数
        # pat_spec_func = re.compile('|'.join([f+'\(.*?\)' for f in SPEC_FUNCS]), re.I)
        pat_spec_func = re.compile('|'.join(SPEC_FUNCS), re.I)
        # 匹配预定义操作函数
        pat_opra_func = re.compile('|'.join(OPRA_FUNCS), re.I)

        while expn:
            # 匹配操作符
            if expn[0] in OPTR_PRI:
                optr, expn = expn[0], expn[1:]
                # 比较操作符优先级
                if optr == '(' or s1.is_empty():
                    s1.push(optr)
                elif optr == ')':
                    while s1.peek() != '(':
                        s2.push(s1.pop())
                    s1.pop()  # 当遇到')'时S1中必定至少有一个对应的'(', 如果pop失败说明表达式不合法
                    # 遇到属于函数的'(', 立即计算函数
                    if not s1.is_empty() and s1.peek() in OPRA_FUNCS:
                        s2.push(s1.pop())
                else:
                    while not s1.is_empty() and OPTR_PRI[s1.peek()] >= OPTR_PRI[optr]:
                        s2.push(s1.pop())
                    s1.push(optr)
            else:
                # 匹配变量操作数入栈, 这里仍然带'[]',作为变量的标识,便于后缀表达式计算
                operand = pat_operand.match(expn)
                if operand:
                    s2.push(operand.group())
                    expn = expn[operand.span()[1]:]
                else:
                    # 匹配数值操作数, 转换为float入栈
                    operand = pat_digital.match(expn)
                    if operand:
                        s2.push(float(operand.group()))
                        expn = expn[operand.span()[1]:]
                    else:
                        # 匹配特殊函数(带额外参数),将函数的返回值入栈
                        operand = pat_spec_func.match(expn)
                        if operand:
                            func_expn = ExpnResolution._search_func_content(operand.group(), expn)
                            rtn = ExpnResolution._execute_function(func_expn)
                            if rtn is not None:
                                s1.push(rtn)
                                expn = expn[len(func_expn):]
                            else:
                                return None  # 函数执行异常
                        else:
                            # 匹配操作函数入栈
                            operand = pat_opra_func.match(expn)
                            if operand:
                                s1.push(operand.group().upper())
                                expn = expn[operand.span()[1]:]
                            else:
                                return None  # 匹配失败

        while not s1.is_empty():
            s2.push(s1.pop())
        return s2

    def _calc_postfix(self, postfix: Stack):
        """
        :param postfix: 后缀表达式栈
        :return: 返回计算结果
        """
        if postfix.is_empty():
            return None

        stack = Stack()
        for x in postfix.items:
            if not isinstance(x, str):
                # 可操作数据, 直接入栈
                stack.push(x)
            elif x[0] == '[':
                # 变量名操作数, 读取数据入栈
                stack.push(self._get_vars(x))
            elif x == '+':
                a, b = stack.pop(), stack.pop()
                stack.push(b + a)
            elif x == '-':
                a, b = stack.pop(), stack.pop()
                stack.push(b - a)
            elif x == '*':
                a, b = stack.pop(), stack.pop()
                stack.push(b * a)
            elif x == '/':
                a, b = stack.pop(), stack.pop()
                stack.push(b / a)
            elif x == '%':
                a, b = stack.pop(), stack.pop()
                stack.push(b % a)
            elif x == '^':
                a, b = stack.pop(), stack.pop()
                stack.push(b ** a)

            elif x.upper() == 'AVG':
                stack.push(np.mean(stack.pop()))
            elif x.upper() == 'SUM':
                stack.push(np.sum(stack.pop()))
            elif x.upper() == 'MAX':
                stack.push(np.max(stack.pop()))
            elif x.upper() == 'MIN':
                stack.push(np.min(stack.pop()))
            elif x.upper() == 'LOG':
                stack.push(np.log10(stack.pop()))
            elif x.upper() == 'INT':
                stack.push(np.int64(stack.pop()))
            elif x.upper() == 'RND':
                stack.push(np.round(stack.pop()))
            elif x.upper() == 'CNT':
                dat = stack.pop()
                if isinstance(dat, pd.Series):
                    stack.push(dat.count())
            elif x.upper() == 'MODE':
                dat = stack.pop()
                if isinstance(dat, pd.Series):
                    stack.push(dat.mode().tolist()[0])
            elif x.upper() == 'UNI':
                dat = stack.pop()
                if isinstance(dat, pd.Series):
                    # 为规避列表的[]和变量名的[]识别混淆, 返回元组
                    return tuple(dat.unique().tolist())
            elif x.upper() == 'PCT':
                dat = stack.pop()
                if isinstance(dat, pd.Series):
                    return dat.value_counts(ascending=False, normalize=True)
            else:
                # TODO 未定义的函数错误
                return False
        return stack.peek()

    @ staticmethod
    def _execute_function(func_expn):
        # 匹配预定义参数函数名
        fc = re.match('(.*?)\((.*)\)', func_expn, re.I)
        if fc:
            func_name, para_str = fc.groups()
            paras = para_str.split(',')
            if paras[-1] == '':
                paras.pop()  # 最后一个符号为','时split会多一个空元素
            dat = ExpnResolution(paras[0]).result

            func_name = func_name.upper()
            if func_name == 'CNTIF':
                if isinstance(dat, pd.Series):
                    if len(paras) > 1:
                        cond = ExpnResolution._check_brief_cond(dat.name, paras[1])
                        if re.search(r'[^\w\'\"\[\]]', cond):  # 含运算符的表达式
                            try:
                                return int(dat.to_frame().query(cond).count())
                            except Exception as e:
                                # 筛选条件表达式无效
                                return None
                        else:
                            counter = dat.value_counts(ascending=False)
                            cond = cond.strip('\'').strip('\"')  # 外部对文本值可能使用了''号或"",这里不需要
                            if cond in counter:
                                return counter[cond]
                            else:
                                return 0
                    else:
                        return dat.value_counts(ascending=False)
                return dat
            if func_name == 'FILTER':
                if isinstance(dat, pd.Series):
                    if len(paras) > 1:
                        cond = ExpnResolution._check_brief_cond(dat.name, paras[1])
                        return dat.to_frame().query(cond)[dat.name]
                    else:
                        return dat
                return dat
            if func_name == 'PERCENT':
                if isinstance(dat, pd.Series):
                    if len(paras) > 1:
                        cond, _ = ExpnResolution._check_cond(paras[1])
                        if re.search(r'[^\w\'\"]', cond):
                            # 是含有运算符的表达式
                            cond, _ = ExpnResolution._check_cond(paras[1])
                            dat.to_frame().query(cond)
                        else:
                            return dat.value_counts(ascending=False, normalize=True)[paras[1]]
                    else:
                        return dat.value_counts(ascending=False, normalize=True)
                return None
            if func_name == 'LARGE':
                if isinstance(dat, pd.Series):
                    if len(paras) > 2:
                        return dat.nlargest(int(paras[1]))
                    else:
                        return list(dat.nlargest(int(paras[1])))[-1]
                return None
            if func_name == 'SMALL':
                if isinstance(dat, pd.Series):
                    if len(paras) > 2:
                        return dat.nsmallest(int(paras[1]))
                    else:
                        return list(dat.nsmallest(int(paras[1])))[-1]
                return None
        else:
            return None


# 定义临时测试数据
kpi1 = pd.DataFrame({'A': [9,1,1,8,2,6,9,9,3,9],
                     'B': [144,367,447,486,224,196,395,458,351,173],
                     'C': [59.4,41.57,14.65,89.82,72.34,67.5,61.16,19.11,42.27,22.63],
                     'D': ['a','f','d','c','b','a','d','b','g','b'],
                     'E': [1,2,0,2,0,0,2,0,0,2]})
kpi2 = pd.DataFrame({'X': [50,74,31,68,78,25,68,73],
                     'Y': [1,0,0,1,0,1,0,0],
                     'Z': ['c','c','a','a','a','a','b','b']})

if __name__ == "__main__":

    """
    :param expn: 计算表达式（中缀表达式)
    表达式规则: 表达式为数值,有效变量,有效运算符,以及有效函数的合法组合
    1. 变量必须使用[]包围
    2. 有效运算符有内置字典MO_PRI定义
    3. 有效函数名由内置列表OPRA_FUNCS定义,函数内容由()包围
    4. 如存在筛选条件,将条件表达式放入{}中
    5. 条件表达式中匹配的文本内容需要使用''或""包围
    6. 如条件表达式内嵌套计算表达式, 使用':'标识起始位置, 注意嵌套计算表达式会被其返回字符串替换, 请确保替换后条件表达式仍然有效
    
    示例1： 计算表达式 '10*LOG([xx跟踪/A列]*2 + [xx跟踪/B列]*(3+1)) * SUM([xx跟踪/C列]))'
    示例2： 有条件的计算表达式 '10*LOG([xx跟踪/A列]*2  {[xx跟踪/B列]=='Pass'}' 注意文本使用''
    示例3： 有嵌套条件的计算表达式 '([xx跟踪/A列]/1000) {[xx跟踪/C列] in :UNI([yy跟踪/A列]  {[yy跟踪/B列]==1}}'
    """

    # 功能测试

    # 组合计算
    print(ExpnResolution('SUM(RND(10 * LOG([kpi1/A] + [kpi1/B]^2 + 2)))').result)

    # 纯数值计算
    print(ExpnResolution('rnd(10 * log(2)) × (5+2) - 6**2').result)

    # 条件筛选
    print(ExpnResolution('(AVG([kpi2/X] + [kpi2/Y] * 3)) {[kpi2/Y] == 1}').result)
    print(ExpnResolution('[kpi2/X] {[kpi2/Z] == \'a\'}').result)

    # 筛选函数与外部条件表达式的区别: 筛选函数的对象是series使用自身条件筛选, 外部表达式可以使用不同列作为筛选条件
    print(ExpnResolution('FILTER([kpi1/A], =9)').result)
    print(ExpnResolution('[kpi1/A]{[kpi1/A] =9}').result)
    print(ExpnResolution('[kpi1/A]{[kpi1/E] =0}').result)

    # 嵌套条件筛选
    print(ExpnResolution('UNI([kpi2/Z]) {[kpi2/Y]==1}').result)  # 嵌套内容
    print(ExpnResolution('[kpi1/C]/  [kpi1/B] {[kpi1/D] in :UNI([kpi2/Z]) {[kpi2/Y]==1}}').result)

    # 特殊函数
    print(ExpnResolution('LARGE([kpi1/C], 3)').result)
    print(ExpnResolution('LARGE([kpi1/C], 3, True)').result)

    print(ExpnResolution('CNTIF([kpi2/Z])').result)
    print(ExpnResolution('CNTIF([kpi1/C], <70&>50)').result)

    print(ExpnResolution('PERCENT([kpi1/D])').result)

```



# 函数统一接口架构

```python
from types import FunctionType, MethodType
from enum import Enum


# 基础类
class FuncClass(object):
    """
    函数类基类，实现将类的实例当做函数来使用
    """
    def __init__(self, name, func):
        # type: (str, (FunctionType, MethodType)) -> None
        """
        构造方法
        :param name: 名称
        :param func: 函数
        """
        self.name = name
        self._func = func

    def __call__(self, *args, **kwargs):
        """
        类的实例可以当函数来使用
        示例: f = FunctionClass('sum', sum)
             print(f([1, 2, 3]))
        等价: print(sum([1, 2, 3]))
        """
        return self._func(*args, **kwargs)


# 函数定义
def func_a1(name, age):
    print('a1: My name is %s, I am %s.' % (name, age))
    return


def func_a2(name, age):
    print('a2: I\'m %s, %s years old.' % (name, age))


def func_b(name, sport):
    print('b: %s like %s.' % (name, sport))


# 函数联想
class _FuncA(Enum):
    # 方便对函数分类和联想(枚举)
    a1 = FuncClass('a1', func_a1)
    a2 = FuncClass('a2', func_a2)


class _FuncB(Enum):
    b = FuncClass('b', func_b)


# 统一入口
def func_access(*args, **kwargs):
    """
    :param args: 分支功能(类）
    :param kwargs: 参数, 对于args里的所有分支都会应用该参数
    :return:
    """
    if not args:
        raise ValueError('未指定分支功能')

    _name = _sport = ''
    _age = 0

    if 'info' in kwargs:
        _name = kwargs['info'][0]
        _age = kwargs['info'][1]
    if 'sport' in kwargs:
        _name = kwargs['sport'][0]
        _sport = kwargs['sport'][1]

    # 函数实例分组
    cate_a, cate_b = [], []
    for _func in args:
        if isinstance(_func, _FuncA):
            cate_a.append(_func)
        elif isinstance(_func, _FuncB):
            cate_b.append(_func)

    if cate_a:
        if _name == '' or _age <= 3:
            # 参数校验
            raise ValueError('No name or too young.')
        for a in cate_a:
            a.value(_name, _age)  # 通过实例执行函数

    if cate_b:
        if _name == '' or _sport == '':
            raise ValueError('No name or No sport.')
        for b in cate_b:
            b.value(_name, _sport)  # 通过实例执行函数


if __name__ == "__main__":
    a1 = _FuncA.a1
    a2 = _FuncA.a2
    b = _FuncB.b

    func_access(a1, info=('Jim', 12))
    func_access(a2, info=('Sam', 11))
    func_access(b, sport=('Ted', 'Swimming'))

```
