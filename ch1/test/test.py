import sys
import os
sys.path.append(os.path.split(__file__)[0] +'/../..')
import ch1 as light
from ch1.backend import backend as np

x1 = light.functions.placeholder()
x2 = light.functions.placeholder()
x3 = light.functions.constant(1.0)
y = light.functions.add(x1, light.functions.add(x2, x3))
forward = light.functions.executor(*[y])
backward = light.functions.gradient(y, *[y, x1, x2, x3])

x1_shape = (2, 3)
x2_shape = (3, )
for i in range(100):
    x1_v = np.random.random(x1_shape)
    x2_v = np.random.random(x2_shape)
    feed_dict = {x1:x1_v, x2:x2_v}
    y_v, = forward.run(feed_dict=feed_dict)
    grad_y, grad_x1, grad_x2, grad_x3 = backward.get()

    print("y")
    print(y_v)
    print("grad_y")
    print(grad_y)
    print("grad_x1")
    print(grad_x1)
    print("grad_x2")
    print(grad_x2)
    print("grad_x3")
    print(grad_x3)
    print("________________________")


"""
L = [x3, x2, y1, x1, y]
feed_dict={x1:1, x2:2}
for node in L:
    if isinstance(node, Placeholder):
        x1.output_value = feed_dict[node]
    else:
        node.compute_output()
L[-1].output_value就是最终y的值
"""
"""
executor(*[y])等同于Executor([y])
"""
"""
x1 = light.functions.placeholder()
x2 = light.functions.placeholder()
x3 = light.functions.constant(1.0)
y1 = light.functions.add(x2, x3)
y = light.functions.add(x1, y2)
forward = light.functions.executor(*[y1, y])
"""