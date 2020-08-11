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

class Placeholder(object):
    def __init__(self):
        self.output_value = None
        self.output_nodes = []

class Variable(object):
    def __init__(self, initial_value): 
        self.output_value = initial_value
        self.output_nodes = []

    def compute_output(self):
        return self.output_value

class Constant(object):
    def __init__(self, value):
        self.output_value = value
        self.output_nodes = []

    def compute_output(self):
        return self.output_value



class Operation(object):
    pass

class ArithmeticalOperation(Operation):
    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        for node in input_nodes:
            node.output_nodes.append(self)

    def compute_output(self):
        raise NotImplementedError

    def gradients_function(self):
        raise NotImplementedError

class LogicalOperation(Operation):
    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        for node in input_nodes:
            node.output_nodes.append(self)

    def compute_output(self):
        raise NotImplementedError

class Add(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            grad_wrt_x = grad
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            y = self.input_nodes[1].output_value
            grad_wrt_y = grad
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y
        
        return [grad_x, grad_y]

class Argmax(LogicalOperation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.argmax(x.output_value, axis=self.axis)
        return self.output_value

class MatMul(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            y = self.input_nodes[1].output_value
            return np.dot(grad, np.transpose(y))

        def grad_y(grad):
            x = self.input_nodes[0].output_value
            return np.dot(np.transpose(x), grad)

        return[grad_x, grad_y]

class Square(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.multiply(2.0, x.output_value))
        
        return [grad_x]

class Mean(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.mean(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            input_value = self.input_nodes[0].output_value
            output_shape = np.array(np.shape(input_value))
            d = np.prod(output_shape[self.axis])
            grad = np.divide(grad, d)
            output_shape[self.axis] = 1
            tile_scaling = np.shape(input_value)//output_shape
            return np.tile(grad, tile_scaling)
        
        return [grad_x]

class Sum(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            input_value = self.input_nodes[0].output_value
            output_shape = np.array(np.shape(input_value))
            output_shape[self.axis] = 1
            tile_scaling = np.shape(input_value)//output_shape
            return np.tile(grad, tile_scaling)
        
        return [grad_x]

class Subtract(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.subtract(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            grad_wrt_x = grad
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x
        
        def grad_y(grad):
            y = self.input_nodes[1].output_value
            grad_wrt_y = np.negative(grad)
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y

        return [grad_x, grad_y]

class Multiply(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = [node.output_value for node in self.input_nodes]
            
            grad_wrt_x = np.multiply(grad, y)
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            x, y = [node.output_value for node in self.input_nodes]
            grad_wrt_y = np.multiply(grad, x)
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y
        return [grad_x, grad_y]

class Divide(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.divide(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = [node.output_value for node in self.input_nodes]
            
            grad_wrt_x = np.multiply(grad, np.divide(1.0, y))
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            x, y = [node.output_value for node in self.input_nodes]
     
            grad_wrt_y = np.negative(np.multiply(grad, np.divide(x, np.square(y))))
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y

        return [grad_x, grad_y]       

class Negative(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.negative(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.negative(grad)
        return [grad_x]

class Sqrt(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sqrt(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.multiply(0.5, np.divide(1.0, self.output_value)))
        
        return [grad_x]

class Power(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y= self.input_nodes
        self.output_value = np.power(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = self.input_nodes
            grad = np.multiply(grad, np.multiply(y.output_value, np.power(x.output_value, np.subtract(y.output_value, 1.0))))
            
            for _ in range(np.ndim(grad) - np.ndim(x.output_value)):
                grad = np.sum(grad, axis=0)
            
            for i, dim in enumerate(np.shape(x.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            
            return grad
        
        def grad_y(grad):
            x, y = self.input_nodes
            grad = np.multiply(grad, np.multiply(np.log(x.output_value), self.output_value))
            
            for _ in range(np.ndim(grad) - np.ndim(y.output_value)):
                grad = np.sum(grad, axis=0)
            
            for i, dim in enumerate(np.shape(y.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
                    
            return grad

        return [grad_x, grad_y]

class Exp(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.exp(x.output_value)
        return self.output_value
        
    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, self.output_value)
        return [grad_x]

class Log(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.log(x.output_value)
        return self.output_value
        
    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            return np.divide(grad, x)
        return [grad_x]

class Tanh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.tanh(x.output_value)
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.subtract(1.0, np.square(self.output_value)))
        return [grad_x]

class Max(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.max(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            if self.keepdims:
                return np.multiply(grad, np.equal(self.output_value, x.output_value))
            else:
                outshape = np.array(np.shape(x.output_value))
                outshape[self.axis] = 1
                t = np.shape(x.output_value)//outshape
                r = np.reshape(self.output_value, outshape)
                g = np.tile(r, t)
                grad = np.reshape(grad, outshape)
                eq = np.equal(g, x.output_value)
                return np.multiply(grad, eq)
        return [grad_x]

from queue import Queue
def output_degree(n, nodes, visited):
    for node in n.output_nodes:
        if node in nodes and node not in visited:
            return 1
    return 0

def _get_prerequisite(operation):

    nodes = set()
    queue = Queue()
    queue.put(operation)
    nodes.add(operation)
    while not queue.empty():
        current = queue.get()

        if isinstance(current, Operation):
            for node in current.input_nodes:
                if node not in nodes:
                    queue.put(node)
                    nodes.add(node)
    
    L = []
    visited = set()
    L.append(operation)
    visited.add(operation)
    queue.put(operation)
    while not queue.empty():
        current = queue.get() 
        if isinstance(current, Operation):
            for node in current.input_nodes:
                if node not in visited:
                    if output_degree(node, nodes, visited) == 0:
                        L.append(node)
                        queue.put(node)
                        visited.add(node)
    return list(reversed(L))

class Executor(object):
    def __init__(self, objectives):
        self.objectives = objectives
        self.postorder_nodes = []
        self.init()
        
    def init(self):
        for operation in self.objectives:
            postorder_nodes = _get_prerequisite(operation)
            self.postorder_nodes.append(postorder_nodes)
        
    def run(self, feed_dict=None):
        computed_set = set()
        results = []
        for postorder_nodes in self.postorder_nodes:
            for node in postorder_nodes:
                if node not in computed_set:                    
                    if type(node) is Placeholder:
                        node.output_value = feed_dict[node]
                    else:
                        node.compute_output()
                    computed_set.add(node)
            results.append(postorder_nodes[-1].output_value)
        return results


def input_degree(n, visited, nodes):
    if isinstance(n, Operation):
        for n0 in n.input_nodes:
            if n0 in nodes:
                if n0 not in visited:
                    return 1
    return 0

def topo_sort(nodes, dx):
    queue = Queue()
    visited = set()
    queue.put(dx)
    visited.add(dx)
    while not queue.empty():
        current = queue.get()
        for node in current.output_nodes:
            if isinstance(node, ArithmeticalOperation):
                if node in nodes and node not in visited:
                    queue.put(node)
                    visited.add(node)
    
    L = []
    current_visited = set()
    queue.put(dx)
    current_visited.add(dx)
    while not queue.empty():
        node = queue.get()
        L.append(node)
        for output_node in node.output_nodes:
            if isinstance(output_node, ArithmeticalOperation):
                if output_node in visited and output_node not in current_visited:
                    degree = input_degree(output_node, current_visited, visited)
                    if degree == 0:
                        queue.put(output_node)
                        current_visited.add(output_node)
    return L

class Gradient(object):
    def __init__(self, objective, variables):
        self.objective = objective #objective 是目标函数#variables 是需要求导的变量的列表
        self.variables = variables
        self.grad_funs = {}
        self.topos = []
        self.can_be_computed = None
        self.init()
    
    def init(self):
        target_op = self.objective
        queue = Queue()
        visited = set()
        queue.put(target_op)        
        visited.add(target_op)
        while not queue.empty():
            node = queue.get()
            if isinstance(node, ArithmeticalOperation):
                for input_node in node.input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.put(input_node)
        self.can_be_computed = visited
        for var in self.variables:
            if var not in visited:
                self.topos.append(None)
            else:
                topo = list(reversed(topo_sort(visited, var)))
                self.topos.append(topo)
                
                for node in topo:
                    if isinstance(node, ArithmeticalOperation):
                        if self.grad_funs.get(node) is None:
                            self.grad_funs[node] = node.gradients_function()

    def get(self, grad=None):
        target_op = self.objective
        grad_table = {}
        if grad is None:
            grad_table[target_op] = np.ones_like(target_op.output_value)
        else:
            grad[target_op] = grad
        results = []
        for i, topo in enumerate(self.topos):
            variable = self.variables[i]
            grad = grad_table.get(variable)
            if grad is None:
                if topo is None:
                    grad = np.zeros_like(variable.output_value)
                    grad_table[variable] = grad
                else:
                    for v in topo:
                        grad_v = grad_table.get(v)
                        if grad_v is None:
                            for output_node in v.output_nodes:
                                if not isinstance(output_node, ArithmeticalOperation):
                                    continue
                                if output_node in self.can_be_computed:
                                    grad_output = grad_table.get(output_node)
                                    funs = self.grad_funs.get(output_node)
                                    grad_v_partial = None
                                    if len(funs) == 1:
                                        grad_v_partial = funs[0](grad_output)
                                    else:
                                        index = output_node.input_nodes.index(v)
                                        grad_v_partial = funs[index](grad_output)
                                    if grad_v is None:
                                        grad_v = grad_v_partial
                                    else:
                                        grad_v = np.add(grad_v, grad_v_partial)
                            grad_table[v] = grad_v
                        grad = grad_v
                    
            results.append(grad)
        return results

class Getitem(ArithmeticalOperation):
    def __init__(self, x, key):
        super(self.__class__, self).__init__(x)
        self.key = key

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = x.output_value.__getitem__(self.key)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            r = np.zeros_like(x.output_value)
            r.__setitem__(self.key, grad)          
            return r
        return [grad_x]

class ExpandDims(ArithmeticalOperation):
    def __init__(self, x, axis=0):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.expand_dims(x.output_value, axis=self.axis)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]

class Concatenate(ArithmeticalOperation):
    def __init__(self, lists, axis=0):
        super(self.__class__, self).__init__(*lists)
        self.axis = axis
        self.indexs = None

    def compute_output(self):
        #x, = self.input_nodes
        lists = [node.output_value for node in self.input_nodes]
        self.output_value = np.concatenate(lists, axis=self.axis)
        return self.output_value

    def gradient_function(self, i):  
        def grad_i(grad):
            if self.indexs is None:
                indexs = []
                axis = self.axis
                begin = 0
                end = begin
                for node in self.input_nodes:
                    shape = np.shape(node.output_value)
                    length = shape[axis]
                    index = [slice(None, None, None) for i in range(len(shape))]
                    end = begin + length
                    index[axis] = slice(begin, end, None)
                    indexs.append(tuple(index))
                    begin = end
                self.indexs = indexs
            return grad.__getitem__(self.indexs[i])
        return grad_i

    def gradients_function(self):
        funs = []
        for i in range(len(self.input_nodes)):
            funs.append(self.gradient_function(i))
        return funs

class Reshape(ArithmeticalOperation):
    def __init__(self, x, new_shape):
        super(self.__class__, self).__init__(x)
        self.new_shape = new_shape

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.reshape(x.output_value, self.new_shape)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]