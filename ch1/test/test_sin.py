import sys
import os
sys.path.append(os.path.split(__file__)[0] +'/../..')
import ch1 as light
from ch1.backend import backend as np

x = light.functions.placeholder()
y = light.functions.sin(x)
forward = light.functions.executor(*[y])
backward = light.functions.gradient(y, *[y, x])

x_v = np.random.rand(2, 3)
feed_dict = {x:x_v}
y_v, = forward.run(feed_dict=feed_dict)
grad_y, grad_x = backward.get()
print(grad_y, grad_x)
