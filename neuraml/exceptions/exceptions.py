__all__ = [
    "NoneError",
    "InstanceNotCalledError",
    "NotInListError",
    "EmptyListError",
    "EmptyDataFrameError",
    "VariableInitializationError",
]


class NoneError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "None Value Assigned To Variable!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class InstanceNotCalledError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Instance not called, Please call the instance() method / __call__ operator!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class NotInListError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Variable Not Present In List!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class EmptyListError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Please Provide Valid List Empty List Found!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class EmptyDataFrameError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Please Provide Valid DataFrame Empty DataFrame Found!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class VariableInitializationError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Variable Not Initialized!"
        else:
            self.msg = msg

        super().__init__(self.msg)
