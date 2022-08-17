# Python概述

[让你用sublime写出最完美的python代码--windows环境](https://www.cnblogs.com/zhaof/p/8126306.html)



Python官网：https://www.python.org/

多Python并存：电脑-属性-高级系统设置-环境变量-PATH，在最上面的路径为模型启动的Python版本

Python模块的下载路径：https://pypi.org/simple/ 

- python文件通常有两种使用方法，第一是作为**脚本直接执行**，第二是 import 到其他的 python 脚本中**被调用（模块重用）**

- 每个python模块（python文件，也就是此处的 test.py 和 import_test.py）都包含**内置的变量 \__name__**，当该模块被**直接执行**的时候，\__name__ 等于文件名（包含后缀 .py ）；如果该模块 import到其他模块中，则该模块的 \__name__ 等于模块名称（不包含后缀.py）。而 “\_\_main__” **始终指当前执行模块的名称**（包含后缀.py）

  ```python
  # test.py，直接执行会获得2行输出
  print("this is func")
  
  if __name__ == '__main__': # 程序入口
  	print("this is main") 
  ```

  ```python
  # test_import.py，调用test.py，只会输出"this is func"
  import test
  ```

- python 中，类型属于对象，变量仅仅是引用（指针）没有类型

  ```python
  a=[1,2,3]
  a="Runoob"
  # 以上代码中，[1,2,3]是List类型，"Runoob"是String类型，而变量a没有类型,仅仅是一个对象的引用（指针）
  ```

- 参数

  ```python
  def func(param = 'name') # 默认参数值
  	pass
  
  def printinfo( arg1, *vartuple ): # 不定长参数使用*号，以元组形式导入（可以为空）
  	for var in vartuple:
        print (var)
  	return
  	
  def printinfo( arg1, *vartuple ): # 不定长参数使用**号，以字典形式导入（可以为空）
  	print (vardict)
  	return
  	
  # lambda [arg1 [,arg2,.....argn]]:expression # 一种简化的函数定义
  sum = lambda arg1, arg2: arg1 + arg2
  
  func(a = "wang") # 可以在调用中赋值
  printinfo(1, a=2,b=3)
  ```

  

---

# 装饰器

python装饰器就是用于**拓展原来函数功能的一种函数**，这个函数的特殊之处在于它的返回值也是一个函数，使用python装饰器的好处就是在不用更改原函数的代码前提下给函数增加新的功能。装饰器本质上是一个高级Python函数，通过给别的函数添加@标识的形式实现对函数的装饰。它经常用于有切面需求的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量与函数功能本身无关的雷同代码并继续重用。

一段原始代码：

```python
import time
def func():
    print("hello")
    time.sleep(1)
    print("world")
```

我们试图记录下这个函数执行的总时间：

```python
#原始侵入（篡改原函数）
import time
def func():
    startTime = time.time()
    print("hello")
    time.sleep(1)
    print("world")
    endTime = time.time()
    msecs = (endTime - startTime)*1000
    print("time is %d ms" %msecs)
```

但是如果不允许篡改原始代码来实现，就可以使用装饰器了：

```python
# 不需要侵入，也不需要函数重复执行
import time
def deco(func):
    def wrapper():
        startTime = time.time()
        func()
        endTime = time.time()
        msecs = (endTime - startTime)*1000
        print("time is %d ms" %msecs)
    return wrapper
# deco函数就是最原始的装饰器，它的参数是一个函数，然后返回值也是一个函数。其中作为参数的这个函数func()就在返回函数wrapper()的内部执行
# 在函数func()前面加上@deco，func()函数就相当于被注入了计时功能
@deco
def func():
    print("hello")
    time.sleep(1)
    print("world")

if __name__ == '__main__':
    f = func #这里f被赋值为func，执行f()就是执行func()
    f()
```

带有不定参数的装饰器：

```python
#带有不定参数的装饰器
import time

def deco(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)*1000
        print("time is %d ms" %msecs)
    return wrapper


@deco
def func(a,b):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b))

@deco
def func2(a,b,c):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b+c))


if __name__ == '__main__':
    f = func
    func2(3,4,5)
    f(3,4)
    #func()
```

多个装饰器执行的顺序是从最后一个装饰器开始，执行到第一个装饰器，再执行函数本身装饰器的加载顺序是从内到外的(从下往上的)。其实很好理解：装饰器是给函数装饰的，所以要从靠近函数的装饰器开始从内往外加载

https://xiaotut.com/54-yy/jisuanji/40322-40322.html

```python
#多个装饰器

import time

def deco01(func):
    print("this is deco01 in")
    def wrapper(*args, **kwargs):
        print("this is deco01")
        startTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        msecs = (endTime - startTime)*1000
        print("time is %d ms" %msecs)
        print("deco01 end here")
    print("this is deco01 out")
    return wrapper

def deco02(func):
    print("this is deco02 in")
    def wrapper(*args, **kwargs):
        print("this is deco02")
        func(*args, **kwargs)

        print("deco02 end here")
    print("this is deco02 out")
    return wrapper

@deco01
@deco02
def func(a,b):
    print("hello，here is a func for add :")
    time.sleep(1)
    print("result is %d" %(a+b))


if __name__ == '__main__':
    f = func
    f(3,4)
    #func()

'''
this is deco01
this is deco02
hello，here is a func for add :
result is 7
deco02 end here
time is 1003 ms
deco01 end here
'''
```



```python
class Animal():
    def __init__(self, kind, age):
        self.kind = kind
        self.age = age

    def info(self):
        print(self.kind, 'age:' + self.age)

class Cat(Animal):
    pass

class Bird(Animal):
    def __init__(self, kind, age, skill):
        Animal.__init__(self, kind, age)
        # self.kind = kind
        # self.age = age
        self.skill = skill

    def info(self):  # 如果在子类中添加一个父类同名的方法，会覆盖父类的方法
        print(self.kind, 'age:' + self.age, 'skill:'+self.skill)

animal = Animal('dog','5')
animal.info()

cat = Cat('cat','3')
cat.info()

bird = Bird('bird', '3', 'fly')
bird.info()
```

# 基础语法

- 指定编码（默认UTF-8）

- 标准数据类型：Number, String, Tuple;   List, Set, Dictionary

  ```python
   type(obj) # 获取类型
   isinstance(obj, type) # 判断类型
  ```

- 注释 # 或 ''' '''

- 多行语句 \ 或 在括号内

- 一行多语句 ; 分隔

- 数字类型 int float bool complex

- 代码组 : 和相同缩进，如if while def class等

- print默认换行输出，不换行可加入参数end print( x, end=" " )

- import 模块，from 模块 import 函数

- 允许多变量赋值：a=b=c=1 或 a,b,c = 10,20,30

   

1. 变量的定义和声明：定义（建立存储空间），声明（不建立存储空间）。变量可以在多个地方声明，但是只能在一个地方定义。
2. 基本类型变量的声明和定义（初始化）是同时产生的；而对于对象来说，声明（创建一个类对象）和定义（类初始化）是分开的。

数据类型：数字(int)、浮点(float)、字符串(str)，列表(list)、元组(tuple)、字典(dict)、集合(set)

```python
def test():
    try:
        print(100) # 一定会执行
    except IndexError as e:
        print(e)  # 如果try中的语句异常则执行这里
    else:
        print(200) # 如果try中的语句正常则执行这里
        return(999) # 函数结束并返回结果
        print(201)
    finally:
        print(666) # 一定会执行
```



动态定义变量

https://segmentfault.com/a/1190000018534188

# 类

```python
class Animal():
    def __init__(self, kind, age):
        self.kind = kind
        self.age = age

    def info(self):
        print(self.kind, 'age:' + self.age)

# 继承
class Bird(Animal):
    pass

# 继承/改写
class Penguin(Bird):
    def __init__(self, kind, age, name):
        Bird.__init__(self, kind, age)  # 可以调用父类的初始化方法，减少代码量
        # Animal.__init__(self, kind, age)

        # self.kind = kind
        # self.age = age
        self.name = name

    def info(self):  # 在子类中添加一个父类同名的方法，会覆盖父类的方法
        print(self.kind, 'age:' + self.age, 'name:'+self.name)

dog = Animal('dog','5')
bird = Bird('bird','3')
penguin = Penguin('penguin', '3', 'Polly')

dog.info()
bird.info()
penguin.info()
```



​	单例模式就是确保一个类只有一个实例。比如某个服务器的配置信息存在在一个文件中，客户端通过AppConfig类来读取配置文件的信息。如果程序的运行的过程中很多地方都会用到配置文件信息，则就需要创建很多的AppConfig实例，造成资源的浪费。其实这个时候AppConfig我们希望它只有一份，就可以使用单例模式。实现单例模式的几种方法如下：

1. 使用模块

   python的模块就是天然的单例模式，因为模块在第一次导入的时候，会生成。pyc文件，当第二次导入的时候，就会直接加载。pyc文件，而不是再次执行模块代码。如果我们把相关的函数和数据定义在一个模块中，就可以获得一个单例对象了。
    新建一个python模块叫singleton，然后常见以下python文件

2. 使用装饰器

   装饰器里面的外层变量定义一个字典，里面存放这个类的实例。当第一次创建的收，就将这个实例保存到这个字典中。
   然后以后每次创建对象的时候，都去这个字典中判断一下，如果已经被实例化，就直接取这个实例对象。如果不存在就保存到字典中。

3. 使用类

   思路就是，调用类的instance方法，这样有一个弊端就是在使用类创建的时候，并不是单例了。也就是说在创建类的时候一定要用类里面规定的方法创建。基于\_\_new\_\_方法实现的单例模式(推荐使用，方便)
    知识点:
    1> 一个对象的实例化过程是先执行类的__new__方法，如果我们没有写，默认会调用object的__new__方法，返回一个实例化对象，然后再调用__init__方法，对这个对象进行初始化，我们可以根据这个实现单例。
    2> 在一个类的__new__方法中先判断是不是存在实例，如果存在实例，就直接返回，如果不存在实例就创建。



工厂模式：factory pattern，所谓“工厂模式”就是专门提供一个“工厂类”去创建对象，我只需要告诉这个工厂我需要什么，它就会返回给我一个相对应的对象。

说的更通俗一点，就是专门创建类型的实例的工厂(类)。





---

# 常用运算

- 运算符

  ```python
  算数运算：+ - * / // % **
  赋值运算：算数运算对应增加=号
  比较运算：== != > < >= <=
  位运算符：& | ^ ~ << >> （二进制）
  逻辑运算：and or not
  成员运算：in not in
  身份运算：is is not
  ```

- 数学函数 **引用模块 import math**

  ```python
  绝对值：abs(x), fabs(x)
  上入整：ceil(x)
  下舍整：floor(x)
  对数：log(100, 10)	== 2
  max(x1, x2,...)
  min(x1, x2,...)
  modf(x)	返回x的小数部分与整数部分（浮点型）
  幂运算：pow(x, y)
  四舍五入：round(x [,n])	
  平方根：sqrt(x)
  三角函数：(a)sin, (a)cos, (a)tan, degrees, radians
  ```

- 随机函数 **引用模块 import random**

  ```python
  choice(seq)	从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。
  randrange ([start,] stop [,step])	从指定范围内，按指定基数递增的集合中获取一个随机数，基数默认值为 1
  random()	随机生成下一个实数，它在[0,1)范围内。
  seed([x])	改变随机数生成器的种子seed。如果你不了解其原理，你不必特别去设定seed，Python会帮你选择seed
  shuffle(lst)	将序列的所有元素随机排序
  uniform(x, y)	随机生成下一个实数，它在[x,y]范围内。
  ```

---

# 字符串



乱码 问题

https://pythontechworld.com/article/detail/fkWl8ZTU70rE

- python 字符串拼接总结

  https://segmentfault.com/a/1190000015475309

```python
import re 
# 去除中文
str1='帮会建了徽信群 没在群里的加下徽信:[30109552300]，晚上群里有活动通知大家，(抢资源)，争地盘，谢谢配合。i love you ' 
linee=re.sub('[\u4e00-\u9fa5]', '', p1)
# 去除标点
simple_punctuation = '[’!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~，。,]'
line = re.sub(simple_punctuation, '', linee)
# 去除数字
re.sub("[0-9]", " ", line)
```

多分隔符分隔

```python
def my_multi_split(s, ds):
    res = [s]
    for d in ds:
        t = []
        list(map(lambda x: t.extend(x.split(d)), res))
        res = t
    return [x for x in res if x]
```

- 单引号和双引号

  ```python
  # 没有任何区别
  str1 = 'python'
  str2 = "python" 
  
  # 单/双引号可以减少转移符的使用，当用单/双引号定义字符串的时候，它就会认为字符串里面的另一种引号是普通字符，从而不需要转义
  str3 = 'We all know that \'A\' and \'B\' are two capital letters.'
  str4_ = "We all know that 'A' and 'B' are two capital letters."
  str5 = 'The teacher said: "Practice makes perfect" is a very famous proverb.'
  
  # 三引号可以实现字符串换行
  str1 = """List of name:
  Hua Li
  Chao Deng
  """
  print(str1)
  # 等效于插入换行符
  str1 = "List of name:\nHua Li\nChao Deng"
  
  三引号也可作为多行注释
  ```

- 字符串 转义符\

  ```python
  r	非转义
  \(在行尾时)	续行符
  \\	反斜杠符号
  \'	单引号
  \"	双引号
  \b	退格(Backspace)
  \000	空
  \n	换行
  \v	纵向制表符
  \t	横向制表符
  \r	回车
  \f	换页
  ```

- 字符运算 r 原始字符串，支持切片

- 多行字符串使用 '''内容'''

- 字符串格式化 %

- 

- https://www.zybuluo.com/kingwhite/note/115240

- 

  ```python
  print ("我叫 %s 今年 %d 岁!" % ('小明', 10))
   %c	 格式化字符及其ASCII码
   %s	 格式化字符串
   %d	 格式化整数
   %u	 格式化无符号整型
   %o	 格式化无符号八进制数
   %x	 格式化无符号十六进制数
   %X	 格式化无符号十六进制数（大写）
   %f	 格式化浮点数字，可指定小数点后的精度
   %e	 用科学计数法格式化浮点数
   %E	 作用同%e，用科学计数法格式化浮点数
   %g	 %f和%e的简写
   %G	 %f 和 %E 的简写
   %p	 用十六进制数格式化变量的地址
   
   f-string 是python3.6之后版本添加,以 f 开头，后面跟着字符串，字符串中的表达式用大括号 {} 包起来
   f'{1+2}' 不用判断格式
  ```

- 内建函数

  ```python
  find(str, beg=0, end=len(string)) 检测 str 是否包含在字符串中，如果指定范围 beg 和 end ，则检查是否包含在指定范围内，如果包含返回开始的索引值，否则返回-1
  rfind(str, beg=0,end=len(string))
  
  replace(old, new [, max]) 将字符串中的 str1 替换成 str2,如果 max 指定，则替换不超过 max 次
  
  len(string)
  join(seq)
  split(str="", num=string.count(str))
  
  isalnum() 如果字符串至少有一个字符并且所有字符都是字母或数字则返 回 True,否则返回 False
  isalpha() 如果字符串至少有一个字符并且所有字符都是字母则返回 True, 否则返回 False
  isdigit() 如果字符串只包含数字则返回 True 否则返回 False
  isdecimal() 检查字符串是否只包含十进制字符，如果是返回 true，否则返回 false
  isnumeric() 如果字符串中只包含数字字符，则返回 True，否则返回 False
  
  title()  返回"标题化"的字符串,就是说所有单词都是以大写开始，其余字母均为小写(见 istitle())
  lower()
  upper()
  swapcase()
  islower()
  isupper()
  
  isspace()
  
  strip([chars]) 在字符串上执行 lstrip()和 rstrip()
  lstrip() 截掉字符串左边的空格或指定字符 （右侧rstrip()）
  
  capitalize() 将字符串的第一个字符转换为大写
  count(str, beg= 0,end=len(string)) 返回 str 在 string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数
  endswith(suffix, beg=0, end=len(string)) 检查字符串是否以 obj 结束
  expandtabs(tabsize=8) 把字符串 string 中的 tab 符号转为空格，tab 符号默认的空格数是 8 
  
  多个替换
  '替换\n 和空格'
  #方法1
  words = words.replace('\n', '').replace(' ', '')
  print(words)
    
  #方法2
  rep = {'\n':'',' ':''}
  rep = dict((re.escape(k), v) for k, v in rep.items())
  #print(rep)
  #print(rep.keys())
  pattern = re.compile("|".join(rep.keys()))
  #print(pattern)
  my_str = pattern.sub(lambda m: rep[re.escape(m.group(0))], words)
  print(my_str)
  #print(words.replace(['\n',' '],''))
  
  ```

  

---

# 列表/元组

列表嵌套层数

```python
def flatten(sequence):
    for item in sequence:
        if type(item) is list:
            for subitem in flatten(item):
                yield subitem
        else:
            yield item
a = [1, 'a', ['b', ['c'], [ ], [3, 4]]]
for x in flatten(a):
    print x,
    
flatten = lambda x: [subitem for item in x for subitem in flatten(item)] if type(x) is list else [x]
```



要求：移除所列表中所有的3, 对列表索引的理解

```python
list_1=[1,2,3,3,3,3,3,3,3,4,5]
print(list_1)

# 方法1
for elm in list_1[::-1]:
    if elm == 3:
        list_1.remove(elm)
print(list_1)

# 方法2
list_1=[1,2,3,3,3,3,3,3,3,4,5]
for elm in list_1[:]:
    if elm == 3:
        list_1.remove(elm)
print(list_1)

# 方法3
list_1=[1,2,3,3,3,3,3,3,3,4,5]
for elm in list_1:
    if elm == 3:
        list_1.remove(elm)
print(list_1)

# 方法4
list_1=[1,2,3,3,3,3,3,3,3,4,5]
for elm in range(len(list_1)):
    if list_1[elm] == 3:
        list_1.pop(-1)
print(list_1)
```





- 基本操作 [定义，访问，变更（元组不能变更），运算]

  ```python
  list1 = [1,2,3,4,5,6,7,8,9] # [元素1,元素2,...]
  list2 = list(range(-9,0)) # list转换函数
  tuple1= (1,2,3,4,5)
  ```

  ```python
  list2[0] = -9 # 修改（下标）
  del list[6] # 删除（下标）
  list1 += list2 # 拼接
  ```

  ```python
  print(list1 + list2) # 连接
  print(list1 * 3) #重复
  for x in list1: print(x, end="/") # 迭代
  ```

  排序

  ```python
  # 两个列表关联排序,sort两个list,先根据第一个list，如果第一个list相同再根据第二个list
  # list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=True)))
  list1, list2 = zip(*sorted(zip(list1, list2)))
  
  # 或
  from operator import itemgetter
  sorta,sortb = [list(x) for x in zip(*sorted(zip(list1, list2), key=itemgetter(0)))]
  
  # 如果需要在第一个list（升序）相同时，根据第二个list降序
  # 因为字符串无法添加负号，改变方向，先对数值型添加负号，再使用reverse=True
  li=[[13, "b"], [10, "b"], [10, "a",], [4,"c"], [1, "d"]]
  sort_li = sorted(li,key=lambda sl: (-sl[0],sl[1]),reverse=True)
  print(sort_li)
  
  # 自定一个函数
  import functools
  def cmp(a, b):
      if b[0] == a[0]:
          if a[1]>b[1]:
              return -1
          else:
              return 1
      if a[0]>b[0]:
          return 1
      else:
          return -1
  li = sorted(li,key=functools.cmp_to_key(cmp))
  print(li)
  
  https://blog.csdn.net/junlee87/article/details/78636053
  ```

  

  列表方法

  ```python
  list1.count(obj) # 出现次数
  list1.index(obj) # 获取索引
  
  list1.append(obj)
  list1.insert(index, obj) # 插入对象
  list1.extend([10,11,12]) # 与 += 运算相同
  
  list1.pop([index=-1]) # 移除索引（返回对象值）
  list1.remove(obj) # 移除对象 或 del list1[1]
  
  list1.reverse() # 逆序 reversed()函数返回一个新列表
  list1.sort(key=None, reverse=False) # 排序 sorted()函数返回一个新列表
  
  list1.clear(); list1.copy()
  ```

- 其他函数/技巧

  ```python
  print(3 in list1) # 判断存在
  len(list1); max(list1); min(list1)
  ```

  ```python
  # map()接收一个函数 f 和一个可迭代对象，并把函数 f 依次作用在 list 的每个元素上，返回一个新的 list
  lst_A = list(map(int,lst_A))
  ```

  ```python
  # 列表推导式：对列表所有元素作用 f 
  [f(elm) for elm in list if ..]
  ```

  ```python
  # 使用zip()可同时遍历两个或更多的序列
  questions = ['name', 'quest', 'favorite color']
  answers = ['lancelot', 'the holy grail', 'blue']
  for q, a in zip(questions, answers):
  	print('What is your {0}?  It is {1}.'.format(q, a))
  ```

  ```python
  # 排序（可以用于所有iterable，如字符串、字典）
  sorted(list,[reverse=True]) # list 的 list.sort() 会修改原始的 list（返回值为None）,改函数返回新列表
  
  example_list = [5, 0, 6, 1, 2, 7, 3, 4]
  result_list = sorted(example_list, key=lambda x: x*-1) # 利用key进行倒序排序
  ```

  切片

```python
# 切片
aList=[3,4,5,6,7,9,11,13,15,17]
print(aList[::])  #[3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
print(aList[::-1])  #[17, 15, 13, 11, 9, 7, 6, 5, 4, 3] # 加负号逆向输出
print(aList[::2])  # [3, 5, 7, 11, 15] # 以步长为2进行输出，输出下标依次为0 2 4 6
print(aList[1::2])  #[4, 6, 9, 13, 17],从一开始步长为2
print(aList[3::])  #[6, 7, 9, 11, 13, 15, 17]
print(aList[3:6])  #省略的是步长，而且不包括下标为6的元素   #[6, 7, 9]

print(List[100]) #IndexError: list index out of range，下标超索引错误
print(List[100:])  #[]，切片不会有下标错误，输出一个空列表

# 利用切片方法实现列表的增加
aList=[3,5,7]
print(aList[len(aList):])   #[]
aList[len(aList):]=[9]  #把原来三位列表的第四位赋值为9
print(aList)  #[3, 5, 7, 9] 

# 利用切片方法实现列表元素的修改
#aList = [3,5,7,9]
aList[:3]=[1,2,3]
print(aList)  #[1, 2, 3, 9]
aList[:3]=[]
print(aList)  #[9]

aList=list(range(10))
print(aList)  #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
aList[::2]=[0]*(len(aList)//2)  #"//"整除
print(aList)  #[0, 1, 0, 3, 0, 5, 0, 7, 0, 9]

# 结合del命令删除列表中的部分元素
aList=[3,5,7,9,11]
del aList[:3]
print(aList)  #[9, 11]

```



---

# 集合

- \# 定义

  ```python
  set1 = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
  set2 = set('abracadabra')
  ```

- \# 运算

  ```python
  if 'Google' in set1 : #  判断
  print(a - b)     # a 和 b 的差集
  print(a | b)     # a 和 b 的并集
  print(a & b)     # a 和 b 的交集
  print(a ^ b)     # a 和 b 中不同时存在的元素
  ```

---

#  字典/集合

- \# 定义

  ```python
  dict1 = {} # 空字典
  dict1['a'] = 10
  dict2 = {'name': 'runoob','code':1, 'site': 'www.runoob.com'}
  dict3 = {x: x**2 for x in (2, 4, 6)}
  ```

- \# 基本操作 

  ```python
  print (dict1['a']) 
  print (dict2.keys())   # 输出所有键
  print (dict2.values()) # 输出所有值
  for k, v in dict1.items(): # 同时获取键和值
  print(k, v)
  
  # 合并字典的方法（字典不支持+运算）
  dict{a, **b} #**可以展开一个字典，执行效率最高
  dict{a.items()+b.items()}
  a.update(b) # 更新了a
  
  
  ```

- 集合定义set，集合与字典的区别，字典是key和value的结对，set可以理解为没有value的字典

  因此set元素同样重复（可以利用set(list)获得不重复的值），

  set运算：a|b 并集，a&b 交集， a-b 差集

分割字典

```
def splitDict(d):
    lists = []
    n = len(d)//2   # length of smaller half
    i = iter(d.items())  # alternatively, i = d.iteritems() works in Python 2
    for x in range(2):
        d = dict(itertools.islice(i, n))  # grab first n items
        lists.append(d)
    return lists
```

---

#  迭代器与生成器

- 迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退

  ```python
  list=[1,2,3,4]
  it = iter(list)    # 创建迭代器对象
  print (next(it))   # 输出迭代器的下一个元素
  
  while True:
      try:
          print (next(it))
      except StopIteration: # StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况
          sys.exit()
  
  for x in it:
      print (x, end=" ") 
      
  class MyNumbers:
    def __iter__(self):
      self.a = 1
      return self
   
    def __next__(self):
      if self.a <= 20:
        x = self.a
        self.a += 1
        return x
      else:
        raise StopIteration
   
  myclass = MyNumbers()
  myiter = iter(myclass)
   
  for x in myiter:
    print(x)
  ```

  





---

# 正则表达式

HTML

https://blog.csdn.net/qq_29883591/article/details/78880352?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control

https://cloud.tencent.com/developer/article/1554371

```python
**引用模块 import re**
```

- ^ 匹配字符串的开始。
- $ 匹配字符串的结尾。
- \* 匹配前面的子表达式任意次
- \+ 匹配前面的子表达式指定次数
- ？ 匹配前面的子表达式一次或0次
- {n} 匹配确定的n次
- \b 匹配一个单词的边界，例如 匹配never中的er，不匹配verb中的er
- \B 匹配非单词边界
- \d 匹配任意数字。
- \D 匹配任意非数字字符。
- x? 匹配一个可选的 x 字符 (换言之，它匹配 1 次或者 0 次 x 字符)。
- x* 匹配0次或者多次 x 字符。
- x+ 匹配1次或者多次 x 字符。
- x{n,m} 匹配 x 字符，至少 n 次，至多 m 次。
- (a|b|c) 要么匹配 a，要么匹配 b，要么匹配 c。
- (x) 一般情况下表示一个记忆组 (remembered group)。你可以利用 re.search 函数返回对象的 groups() 函数获取它的值。
- 正则表达式中的点号通常意味着 “匹配任意单字符”

https://blog.csdn.net/caojinfei_csdn/article/details/86495013

http://blog.sina.com.cn/s/blog_6dc145220100zoe2.html

**Python中re的match、search、findall、finditer区别**

- re.match(pattern, string[, flags])

  从首字母开始开始匹配，string如果包含pattern子串，则匹配成功，返回Match对象，失败则返回None，若要完全匹配，pattern要以$结尾

- re.search(pattern, string[, flags])

  若string中包含pattern子串，则返回Match对象，否则返回None，注意，如果string中存在多个pattern子串，只返回第一个

- re.findall(pattern, string[, flags])

  返回string中所有与pattern相匹配的全部字串，返回形式为数组

- re.finditer(pattern, string[, flags])

  返回string中所有与pattern相匹配的全部字串，返回形式为迭代器

交互顺序

sys.stdout.write(re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})',r'\1,\2,\3',line))

```python
# 编译正则表达式，生成一个正则表达式（ Pattern ）对象，供 match() 和 search() 这两个函数使用
re.compile(pattern[, flags])

pattern = re.compile(r'([a-z]+) ([a-z]+)', re.I)
m = pattern.match('Hello World Wide Web') 
m.groups() # 查看所有分组
m.group(1) # 返回分组1子串
m.span(1) # 返回分组1索引
m.start() 
m.end() 

# re.match 只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回 None，而 re.search 匹配整个字符串，直到找到一个匹配。
re.match(pattern, string, flags=0) # flag修饰符
re.search(pattern, string, flags=0)
 
line = "Cats are smarter than dogs"

# 可以使用()标示出要提取的子串，通过group()提取
matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I) # .* 表示任意匹配除换行符（\n、\r）之外的任何单个或多个字符
if matchObj:
   print ("matchObj.group(0) : ", matchObj.group(0)) # group(0)是匹配到的原始字符串
   print ("matchObj.group(1) : ", matchObj.group(1))
   print ("matchObj.group(2) : ", matchObj.group(2))
else:
   print ("No match!!")

# 替换
re.sub(pattern, repl, string, count=0, flags=0) #  替代内容repl可以是函数


# 功能：匹配出一个字符串中所有的数值（可能含负数、小数），以列表返回
re.findall(r'\-?\d+\.?\d*',str_in) 
re.finditer # 返回迭代器

在正则中，使用.*可以匹配所有字符，其中.代表除\n外的任意字符，*代表0-无穷个,比如说要分别匹配某个目录下的子目录:
match = re.match(r"/(.*)/(.*)/(.*)/", "/usr/local/bin/")


```

---

# 多线程

```python
**引用模块 import threading as thrd**
```

- 	https://www.runoob.com/python3/python3-multithreading.html

```python
#!/usr/bin/python3

import threading
import time

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print ("开始线程：" + self.name)
        print_time(self.name, self.counter, 5)
        print ("退出线程：" + self.name)

def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

# 创建新线程
thread1 = myThread(1, "Thread-1", 1)
thread2 = myThread(2, "Thread-2", 2)

# 开启新线程
thread1.start()
thread2.start()
thread1.join() # 等待至线程中止(结束/异常/超时)
thread2.join()
print ("退出主线程")
```

# 推导式

https://blog.csdn.net/yjk13703623757/article/details/79490476

https://blog.csdn.net/weixin_43790276/article/details/90247423

# 文件操作

## 控制office系列

https://cloud.tencent.com/developer/article/1661483



## 临时保存变量

```python
# f = open(r'D:\pydump2.pckl', 'wb')
# pickle.dump(uni_att, f)
# f = open(r'D:\pydump2.pckl', 'rb')
# uni_att = pickle.load(f)
# f.close()
```

## 类型：txt

python常用的读取文件函数有三种read()、readline()、readlines() 

```python
# read()一次性读全部内容
with open("test.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件
    print(data)

# readline()读取第一行内容
with open("test.txt", "r") as f:
    data = f.readline()
    print(data)
    
# readlines() 列表，一般后续会配合for in使用
# readlines会读到换行符，可用如下方法去除
with open("test.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        print(line)

# 写入文件
with open("test.txt","w") as f:
    f.write("这是个测试！")  # 自带文件关闭功能，不需要再写f.close()
```



```python
import sys

result = []
with open('accounts.txt', 'r') as f:
    for line in f:
        result.append(list(line.strip('\n').split(',')))
print(result)
```

读写模式模式的区别：

r :  读取文件，若文件不存在则会报错

w:  写入文件，若文件不存在则会先创建再写入，会覆盖原文件

a :  写入文件，若文件不存在则会先创建再写入，但不会覆盖原文件，而是追加在文件末尾

rb,wb： 分别于r,w类似，但是用于读写二进制文件

r+ :  可读、可写，文件不存在也会报错，写操作时会覆盖

w+ :  可读，可写，文件不存在先创建，会覆盖

a+ : 可读、可写，文件不存在先创建，不会覆盖，追加在末尾

```python
# 利用While循环对每行的处理

txt_tables = []
f = open("C:/foo.txt", "r",encoding='utf-8')
line = f.readline() # 读取第一行
while line:
    txt_data = eval(line) # 可将字符串变为元组
    txt_tables.append(txt_data) # 列表增加
    line = f.readline() # 读取下一行
print(txt_tables)
f.close()
```

## ASCII码读写

```python
# -*- coding: utf-8 -*-
import numpy as np


def hex2dec(x):
    return int(x, 16)


def read_as_hex(fname):
    data = []  # 按十六进制读取文本文件字节进行保存
    with open(fname, 'rb') as f:
        while True:  # 可以取得每一个十六进制
            c = f.read(1)  # read（1）表示每次读取一个字节长度的数值
            if not c:
                break
            else:
                if ord(c) <= 15:
                    data.append(('0x0' + hex(ord(c))[2:])[2:])  # ord()返回ASCII，前面不加0x0会变成7而不是07
                else:
                    data.append((hex(ord(c)))[2:])  # 获得十六进制的字符串表示
    f.close
    return data


def write_as_hex(fname, data):
    f = open(fname, "w")
    for c in data:
        if c != '0a':
            f.write(chr(int(c, 16)))
    f.close


def complement_code():
    """
    数值由16位二进制表示，高低位进行交换操作，且为补码表示
    例：4C F7 表示的是F7 4C, 将F7 4C表示为二进制，由于最高位为1 所以是负数
    F7 4C 取反加1 再转换为十六进制为 08 B4 十进制为 2228，那么最终就是 -2228
    """
    b0 = '4c'
    b1 = 'f7'
    code = b1 + b0  # 十六进制的补码转换
    if int(code[0], 16) > 7:
        val = ((int(code, 16) ^ int('ffff', 16)) + 1) * (-1)
    else:
        val = int(code, 16)
    return val



def dec2hex_byte(x):
    h = str(hex(x))[2:]
    if len(h) % 2 == 1:
        h = '0' + h
    return h


def my_hex_encode(arr_x):
    arr_b = list(map(hex2dec, arr_x))
    arr_b.append(len(arr_b))
    if len(arr_b) % 2 == 1:
        arr_b.append(0)
    arr_b0, arr_b1 = np.array(arr_b[:int(len(arr_b)/2)]), np.array(arr_b[:int(len(arr_b)/2)-1:-1])
    arr_c = np.vectorize(complex)(arr_b0/256, arr_b1/256)
    return arr_c


def my_hex_decode(arr_c):
    arr_r0, arr_r1 = np.real(arr_c) * 256, np.imag(arr_c) * 256
    arr_r = np.int16(np.hstack((arr_r0, arr_r1[::-1])))
    if arr_r[-1] == 0:
        arr_r = list(arr_r[: -2])
    else:
        arr_r = list(arr_r[: -1])
    arr_r = list(map(dec2hex_byte, list(arr_r)))
    return arr_r


if __name__ == "__main__":
    arr_b = read_as_hex(r'D:\temp\ota_h\test.txt')

    # encode
    arr_c = my_hex_encode(arr_b)
    np.savetxt(r'D:\temp\ota_h\test.csv', arr_c, delimiter=",")

    # decode
    arr_c = np.loadtxt(r'D:\temp\ota_h\test.csv', delimiter=",")
    arr_r = my_hex_decode(arr_c)

    print(arr_b)
    print(arr_r)
    print(len(arr_b), len(arr_r))

    write_as_hex(r'D:\temp\ota_h\test2.txt', arr_r)
```

# 模块/函数



## 模块的调用

https://blog.csdn.net/m0_47670683/article/details/108989698



## tKinter GUI

传递

https://blog.csdn.net/weixin_34168700/article/details/90093476



python一句话之利用文件对话框获取文件路径:https://blog.csdn.net/shawpan/article/details/78759199

https://www.cnblogs.com/shwee/p/9427975.html#D13

选择路径

```python
   def tk_select_path():
    f_path = filedialog.askdirectory()
    tk_path.set(f_path)
    
   root = tk.Tk()
    # root.withdraw()
    tk_path = tk.StringVar()
    # f_path = filedialog.askdirectory()
    #
    # tk_path.set(f_path)

    tk.Label(root, text="目标路径:").grid(row=0, column=0)
    tk.Entry(root, textvariable=tk_path).grid(row=0, column=1)
    tk.Button(root, text="路径选择", command=tk_select_path).grid(row=0, column=2)

    f_path = root.mainloop()
```



## 调试日志logging

## 如何调试 RuntimeWarning ？

https://www.jianshu.com/p/907107c7173d

```python
from numpy import seterr
seterr(all='raise')
# 或
from warnings import simplefilter
simplefilter('error')

# try捕获
```

https://www.cnblogs.com/yyds/p/6901864.html



python中print打印显示颜色 https://blog.csdn.net/qq_34857250/article/details/79673698

\ 033 [显示方式;字体色;背景色m ...... [\ 033 [0m]



## 进制转换

```python
base = [str(x) for x in range(10)] + [chr(x) for x in range(ord('A'), ord('A')+6)]


# 二进制 to 十进制: int(str,n=10)
def bin2dec(string_num):
    return str(int(string_num, 2))


# 十六进制 to 十进制
def hex2dec(string_num):
    return str(int(string_num.upper(), 16))


# 十进制 to 二进制: bin()
def dec2bin(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 2)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])


# 十进制 to 十六进制: hex()
def dec2hex(string_num):
    num = int(string_num)
    mid = []
    while True:
        if num == 0: break
        num, rem = divmod(num, 16)
        mid.append(base[rem])
    return ''.join([str(x) for x in mid[::-1]])


# 十六进制 to 二进制: bin(int(str,16))
def hex2bin(string_num):
    return dec2bin(hex2dec(string_num.upper()))


# 二进制 to 十六进制: hex(int(str,2))
def bin2hex(string_num):
    return dec2hex(bin2dec(string_num))


print(hex2dec('3938'))


# 指定位数2进制
def int2bin(n, count=24):
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])
```

[Python里三个好用的调试神器]: http://www.mamicode.com/info-detail-2808051.html

```python
def dec_integer_transform(m, n):
    int_n = []
    i = 0
    while m > 0:
        q = m % (n ** (i + 1))
        int_n.append(int(q / n ** i))
        m -= q
        i += 1
    return ''.join(map(str, int_n[::-1]))


def dec_decimal_transform(m, n, digit_num=5):
    digit_n = []
    for i in range(digit_num):
        d = m * n
        q = int(d)
        m = d - q
        digit_n.append(q)
    return '0.' + ''.join(map(str, digit_n))
```









## PySnooper

调试模块

```python
pip install pysnooper
```

以装饰器的形式使用该工具，其包含了四个参数:
1、output参数。该参数指定函数运行过程中产生的中间结果的保存位置，若该值为空，则将中间结果输出到控制台。
2、variables参数。该参数是vector类型, 因为在默认情况下，装饰器只跟踪局部变量，要跟踪非局部变量，则可以通过该字段来指定。默认值为空vector。
3、depth参数。该参数表示需要追踪的函数调用的深度。在很多时候，我们在函数中会调用其他函数，通过该参数就可以指定跟踪调用函数的深度。默认值为1。
4、prefix参数。该参数用于指定该函数接口的中间结果前缀。当多个函数都使用的该装饰器后，会将这些函数调用的中间结果保存到一个文件中，此时就可以通过前缀过滤不同函数调用的中间结果。默认值为空字符串。

output 参数使用，运行该代码后，结果会输出到./log/debug.log：

```python
import pysnooper

def add(num1, num2):
    return num1 + num2

@pysnooper.snoop("./log/debug.log", prefix="--*--")
def multiplication(num1, num2):
    sum_value = 0
    for i in range(0, num1):
        sum_value = add(sum_value, num2)
    return sum_value

value = multiplication(3, 4)
```

variables参数使用，需要查看局部变量以外变量时，通过variables参数将需要查看类实例的变量self.num1, self.num2, self.sum_value作为当参数传入snoop的装饰器中：

```python
import pysnooper

class Foo(object):
    def __init__(self):
        self.num1 = 0
        self.num2 = 0
        self.sum_value = 0

    def add(self, num1, num2):
        return num1 + num2
    @pysnooper.snoop(output="./log/debug.log", variables=("self.num1", "self.num2", "self.sum_value"))
    def multiplication(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        sum_value = 0
        for i in range(0, num1):
            sum_value = self.add(sum_value, num2)
        self.sum_value = sum_value
        return sum_value

foo = Foo()
foo.multiplication(3, 4)
```

depth参数使用：

```python
import pysnooper

def add(num1, num2):
    return num1 + num2

@pysnooper.snoop("./log/debug.log", depth=2)
def multiplication(num1, num2):
    sum_value = 0
    for i in range(0, num1):
        sum_value = add(sum_value, num2)
    return sum_value

value = multiplication(3, 4)
```

prefix参数使用，为中间结果打印增加一个前缀，以区分不同的函数调用：

```python
import pysnooper

def add(num1, num2):
    return num1 + num2

@pysnooper.snoop(prefix="我的函数输出__")
def multiplication(num1, num2):
    sum_value = 0
    for i in range(0, num1):
        sum_value = add(sum_value, num2)
    return sum_value

value = multiplication(3, 4)
```
