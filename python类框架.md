# 基类

```python
__all__ = ['ClassType', 'BaseItem', 'BaseJson', 'BaseThread', "BaseSingleton", "BaseFunctionClass"]


class _A:
    pass


"""类类型"""
ClassType = type(_A)


class BaseItem(object):
    """
    自定义枚举类的枚举项基类，其比较是比较的枚举名，而不是其值
    """
    def __init__(self, name, value=''):
        """
        枚举类的枚举名与枚举值
        :param name: 枚举名
        :param value: 枚举名对应的值，默认和name相等
        """
        self.name = name
        self.value = name
        if value:
            self.value = value

    def __repr__(self):
        return '{}.{}'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, BaseItem):
            return self.name == other.name
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, BaseItem):
            return self.name > other.name
        else:
            raise TypeError("'>' not supported between instances of 'str' and '%s'" % type(other))

    def __lt__(self, other):
        if isinstance(other, BaseItem):
            return self.name < other.name
        else:
            raise TypeError("'<' not supported between instances of 'str' and '%s'" % type(other))

    def __ge__(self, other):
        if isinstance(other, BaseItem):
            return self.name >= other.name
        else:
            raise TypeError("'>=' not supported between instances of 'str' and '%s'" % type(other))

    def __le__(self, other):
        if isinstance(other, BaseItem):
            return self.name <= other.name
        else:
            raise TypeError("'<=' not supported between instances of 'str' and '%s'" % type(other))

    def __ne__(self, other):
        return not self == other


class BaseJson(object):
    """
    支持转JSON的基类
    """
    def __init__(self, *args, **kwargs):
        """
        构造方法
        :param args:
        :param kwargs:
        """
        # 类的构造方法的参数字典，必须填写这个才能支持转JSON
        self._construct_paras = [args, kwargs]

    def encode(self):
        """
        将类转换为JSON字典
        :return:
        """
        return {'__class__': {'module_name': self.__module__, 'class_name': self.__class__.__name__,
                              'construct_paras': self._construct_paras}}


class BaseThread(Thread):
    """
    线程基类
    """
    def __init__(self, identity=''):
        # type: (str) -> None
        """
        构造方法
        :param identity: 线程名
        """
        if identity and identity != self.__class__.__name__:
            _name = '{}-{}'.format(self.__class__.__name__, identity)
        else:
            _name = self.__class__.__name__
        super(BaseThread, self).__init__(name=_name)
        self._running = True    # 用于控制线程开始或者结束
        # 记录程序运行状态，heartbeat: 是否成功心跳一次，running：是否正在运行，可以自行添加
        self._status = {"running": True}

    def stop(self):
        """
        停止线程
        :return:
        """
        if self._status["running"]:
            tc.logReport('停止线程[%s]' % self.name)
            self._set_running(False)

    def _set_running(self, running):
        # type: (bool) -> None
        """
        设置运行状态
        :param running:
        :return:
        """
        self._running = running
        self._status["running"] = running

    @property
    def is_running(self):
        # type: () -> bool
        """
        线程是否在执行中。可以用is_alive()检查线程是否存活
        :return:
        """
        return self._status["running"]


class BaseSingleton(object):
    """
    单例模式基类
    """
    _instance = None
    _instance_lock = Lock()

    def __init__(self):
        """
        空的构造方法
        """
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            setattr(cls, "_instance", None)
        if not cls._instance:
            with cls._instance_lock:
                cls._instance = super(BaseSingleton, cls).__new__(cls)
        return cls._instance


class BaseFunctionClass(object):
    """
    函数类基类，可以让类的实例当函数来使用
    """
    def __init__(self, name, func, trace_type=-1):
        # type: (str, (FunctionType, MethodType), int) -> None
        """
        构造方法
        :param name: 名称
        :param func: 函数
        :param trace_type: 跟踪类型ID，只有在需要保存为.tmf时才需要用，可以不传。
                           参考《5G RAN V100R005C10 跟踪监控接口说明书》、《eRAN V100R018C10 跟踪监控接口说明书》
                           给5G分配256个；0x0401-0x0500，对应跟踪消息上报接口中的TrcType字段
        """
        self.name = name
        self._func = func
        self.trace_type = trace_type

    def __call__(self, *args, **kwargs):
        """
        类的实例可以当函数来使用
        示例:
        f=BaseFunctionClass("sum", sum)
        print(f([1, 2, 3]))
        等价于print(sum([1, 2, 3]))
        :param args:
        :param kwargs:
        :return:
        """
        return self._func(*args, **kwargs)

    def equal_name(self, obj):
        """
        判断两个BaseFunctionClass的名称是否相同
        :param obj:
        :return:
        """
        if isinstance(obj, BaseFunctionClass):
            if self.name == obj.name:
                return True
        return False

```



将字典设置为类的属性（动态）

```python
class Sample:
    def _set_attrs(self, params):
        # type: (dict) -> None
        """
        将params的key-value设置为self的属性以及属性的值
        :param params:
        :return:
        """
        for _param_name in params:
            # 处理_param_name，如果_param_name包含类型信息，则将参数值转为对应的类型
            _param_value = str(params[_param_name]).strip()
            _param_name = _param_name.strip()
            if ":" not in _param_name:
                setattr(self, _param_name, _param_value)
            else:
                _param_name, _param_type = re.split(" *: *", _param_name, maxsplit=1)
                _param_type = _param_type.lower()
                if not _param_value:
                    # 保证所有用例参数一样，空值也处理下
                    if _param_type == "list":
                        setattr(self, _param_name, [])
                    elif _param_type == "dict":
                        setattr(self, _param_name, {})
                    else:
                        setattr(self, _param_name, "")
                    continue
                if _param_type == "list":
                    setattr(self, _param_name, self._to_list(_param_value))
                elif _param_type == "dict":
                    setattr(self, _param_name, self._to_dict(_param_value))
                elif _param_type == "int":
                    # int类型的数用pandas读取后会变成浮点类型，前面又转为了浮点格式的字符串
                    setattr(self, _param_name, int(float(_param_value)))
                elif _param_type == "float":
                    setattr(self, _param_name, float(_param_value))
                elif _param_type == "bool":
                    setattr(self, _param_name, bool(_param_value))
                elif _param_type == "str":
                    setattr(self, _param_name, str(_param_value))
                elif _param_type == "func" or _param_type == "eval":
                    setattr(self, _param_name, eval(_param_value))
                else:
                    setattr(self, _param_name, _param_value)
    @staticmethod
    def _to_list(s):
        # type: (str) -> List[str]
        """
        将形式为"1, 3, 4, 5"的字符串转为字符串列表["1", "3", "4", "5"]，逗号前后的空格会自动忽略，其余位置的空格不会忽略
        列表中的空元素会被过滤掉，比如"1, 2, ,"会被转换为["1", "2"]
        :param s:
        :return:
        """
        return list(filter(lambda x: x, re.split(" *, *", s)))

    @staticmethod
    def _to_dict(s):
        # type: (str) -> Dict[str, str]
        """
        将形式为"a:1, b:2"的字符串转为字典["a": "1", "b": "2"]，逗号和冒号前后的空格会自动忽略，其余位置的空格不会忽略
        空元素以及不含冒号的元素会被过滤掉，多个冒号时只会取第1个，比如"a:1:2"会被转换为{"a": "1:2"}
        :param s:
        :return:
        """
        if s.startswith("{") and s.endswith("}"):
            s = s[1:-1]
        _items = list(filter(lambda x: ":" in x, re.split(" *, *", s)))
        return dict([re.split(" *: *", _item, maxsplit=1) for _item in _items])
```

