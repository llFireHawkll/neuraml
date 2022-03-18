__all__ = [
    "NoneException",
    "NotInListException",
    "EmptyDataFrameException",
    "VariableInitializationException",
]


class NoneException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "None Value Passed!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class NotInListException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Variable Not Present In List!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class EmptyDataFrameException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Please Provide Valid DataFrame Empty DataFrame Found!"
        else:
            self.msg = msg

        super().__init__(self.msg)


class VariableInitializationException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            self.msg = "Variable Not Initialized!"
        else:
            self.msg = msg

        super().__init__(self.msg)
