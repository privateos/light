"""
    light--a deep learning framework based on numpy/cupy for the purpose of education
    Copyright (C) 2020  privateos

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

from .operations import MatMul
def matmul(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return MatMul(x, y)

from .operations import Square
def square(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Square(x)

from .operations import Mean
def mean(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Mean(x, axis=axis, keepdims=keepdims)

from .operations import Sum
def sum(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sum(x, axis=axis, keepdims=keepdims)

from .operations import Subtract
def subtract(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Subtract(x, y)

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