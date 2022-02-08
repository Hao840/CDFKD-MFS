from typing import Union, List

__all__ = ['MultiMeter', 'AverageMeter']


class MultiMeter:
    """@DynamicAttrs"""

    def __init__(self):
        self.objs = []

    def register(self, obj: Union[str, List[str]]):
        if isinstance(obj, list):
            for obj_i in obj:
                self.register(obj_i)
        elif isinstance(obj, str):
            self.__setattr__(obj, AverageMeter())
            self.objs.append(self.get(obj))
        else:
            raise RuntimeError(f"Unrecognized input {obj}, require 'str' or list of 'str'.")

    def reset(self):
        for obj in self.objs:
            obj.reset()

    def update(self, obj: str, val, n=1):
        self.get(obj).update(val, n)

    def __getitem__(self, item):
        return self.get(item)

    def get(self, obj: str):
        return self.__getattribute__(obj)

    def remove_all(self):
        self.objs = []


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
