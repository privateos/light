import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/../..')
import ch4 as light
import ch4.backend as np


from ch4.functions import multiply, mean, sum, negative, max, subtract, exp, log
from ch4.functions import variable, matmul, add, getitem, tanh, expand_dims, concatenate

def log_softmax(x):
    m = max(x, axis=-1, keepdims=True)
    y = subtract(x, m)
    e = exp(y)
    s = sum(e, axis=-1, keepdims=True)
    l = log(s)
    return subtract(y, l)

def softmax_crossentropy(y, label):
    l = log_softmax(y)
    m = multiply(l, label)
    n = mean(m, axis=0)
    p = sum(n)
    return negative(p)

def get_variable(shape, dtype):
    return np.random.uniform(-0.1, 0.1, shape).astype(dtype)

def get_bais(shape, dtype):
    return np.zeros(shape, dtype=dtype)


"""
h_t = tanh(h_t_1@U + X_t@W + b)
"""
def uniform(shape, dtype):
    return np.random.uniform(-0.1, 0.1, shape).astype(dtype)
def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)
#28, 28, 20
def simple_rnn(x, time_step, input_features, output_features, dtype=np.float32, init=uniform):
    output = []
    bias = zeros
    U = variable(init((output_features, output_features), dtype=dtype))
    W = variable(init((input_features, output_features), dtype=dtype))
    b = variable(bias((output_features, ), dtype=dtype))
    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    t = tanh(add(matmul(X_t, W), b))
    output.append(expand_dims(t, axis=1))

    h_t_1 = t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        t = tanh(add(matmul(h_t_1, U), add(matmul(X_t, W), b)))
        output.append(expand_dims(t, axis=1))
        h_t_1 = t

    y = concatenate(output, axis=1)#(batch_size, time_step, output_features)
    return U, W, b, y

class SimpleRNN:
    def __init__(self, lr=0.06):
        self.x = light.functions.placeholder()#(N, 28, 28)
        self.label = light.functions.placeholder()#(N, 10)

        U, W, b, y = simple_rnn(self.x, 28, 28, 20, dtype=np.float64)#(N, 28, 20)
        self.U = U
        self.W = W
        self.b = b
        #self.rnn = y
        rnn = light.functions.reshape(y, (-1, 560))

        self.out_w = light.functions.variable(get_variable((560, 10), dtype=np.float64))
        self.out_b = light.functions.variable(get_bais((10, ), dtype=np.float64))
        o = light.functions.matmul(rnn, self.out_w)
        o = light.functions.add(o, self.out_b)


        self.objective = softmax_crossentropy(o, self.label)
        self.y = o

        self.compute_forward_train = light.functions.executor(self.objective)
        self.compute_backward_train = light.functions.gradient(self.objective, *[self.U, self.W, self.b, self.out_w, self.out_b])

        self.compute_forward_test = light.functions.executor(self.y)

        def fn(ins, d):
            ins.output_value -= d

        self.fn = fn
        self.lr = lr

    def forward(self, feed_dict):
        return self.compute_forward_train.run(feed_dict)
    
    def backward(self):
        return self.compute_backward_train.get()

    def update(self, grads):
        lr = self.lr
        self.fn(self.U, lr*grads[0])
        self.fn(self.W, lr*grads[1])    
        self.fn(self.b, lr*grads[2])  
        self.fn(self.out_w, lr*grads[3])
        self.fn(self.out_b, lr*grads[4])

    def predict(self, x):
        feed_dict = {self.x:x}
        r, = self.compute_forward_test.run(feed_dict)
        return r

    def fit(self, x, y, epochs=10, batch_size=64, file=sys.stdout):
        nums = x.shape[0]
        for i in range(epochs):
            begin = 0
            end = batch_size
            loss = 0.0
            while end < nums:
                x_train = x[begin:end]
                y_train = y[begin:end]
                begin = end
                end = end + batch_size
                feed_dict = {self.x:x_train, self.label:y_train}
                cur_loss, = self.forward(feed_dict)
                grads = self.backward()
                self.update(grads)

                loss += cur_loss/nums
            print('epochs:'+str(i+1)+'  loss:'+str(loss), file=file)



file_name = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'mnist.npz')
data = np.load(file_name)
x, y = data['x'], data['y']
x = x/255.0
x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))

x_train, x_test = x[0:60000], x[60000:]
y_train, y_test = y[0:60000], y[60000:]

model = SimpleRNN(lr=0.06)
model.fit(x_train, y_train, epochs=20, batch_size=64)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=-1)
acc = pred == y_test
print("accuracy = " + str(acc.mean()))