"""
light  Copyright (C) 2020  privateos
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c' for details.
"""
from .backend import backend as np

from .operations import Placeholder
from .operations import Variable
from .operations import Constant
from .operations import Add
from .operations import Argmax

from .operations import Executor
from .operations import Gradient

def as_ndarray(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def is_placeholer(x):
    return isinstance(x, Placeholder)

def is_constant(x):
    return isinstance(x, Constant)

def is_variable(x):
    return isinstance(x, Variable)

from .operations import Operation
def is_operation(x):
    return isinstance(x, Operation)

from .operations import ArithmeticalOperation
def is_arithmetical_operation(x):
    return isinstance(x, ArithmeticalOperation)

from .operations import LogicalOperation
def is_logical_operation(x):
    return isinstance(x, LogicalOperation)

def is_light(x):
    return is_placeholer(x) or is_constant(x) or is_variable(x) or is_operation(x)
#################################################################################
def placeholder():
    return Placeholder()

def variable(initial_value):
    return Variable(as_ndarray(initial_value))

def constant(value):
    return Constant(as_ndarray(value))

def add(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Add(x, y)

def argmax(x, axis=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Argmax(x, axis)

def executor(*objectives):
    for n in objectives:
        if not is_light(n):
            raise TypeError('an node in objectives is not light type')
    return Executor(objectives)

def gradient(objective, *variables):
    if not is_light(objective):
        raise TypeError('objective is not light type')
    for n in variables:
        if not is_light(n):
            raise TypeError('an node in variables is not light type')
    return Gradient(objective, variables)