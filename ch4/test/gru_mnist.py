import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/../..')
import ch4 as light
import ch4.backend as np

from ch4.functions import multiply, mean, sum, negative, max, subtract, exp, log
from ch4.functions import variable, matmul, add, getitem, tanh, sigmoid, expand_dims, concatenate, multiply, constant

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
def uniform(shape, dtype):
    return np.random.uniform(-0.1, 0.1, shape).astype(dtype)
def zeros(shape, dtype):
    return np.zeros(shape, dtype=dtype)

#x.shape = (batch_size, time_step, input_features)
#output.shape = (batch_size, time_step, output_features)
"""
z_t = sigmoid(X_t@W_z + h_t_1@U_z)
r_t = sigmoid(X_t@W_r + h_t_1@U_r)
h_t_hat = tanh(X_t@W_t + (r_t*h_t_1)@U_t)
h_t = (1-z_t)*h_t_1 + z_t*h_t_hat
"""
def gru(x, time_step, input_features, output_features, dtype=np.float32, init=uniform):
    output = []

    w_shape = (input_features, output_features)
    u_shape = (output_features, output_features)

    W_z = variable(init(w_shape, dtype=dtype))
    U_z = variable(init(u_shape, dtype=dtype))
    W_r = variable(init(w_shape, dtype=dtype))
    U_r = variable(init(u_shape, dtype=dtype))
    W_t = variable(init(w_shape, dtype=dtype))
    U_t = variable(init(u_shape, dtype=dtype))
    const_1 = constant(np.array(1.0, dtype=dtype))

    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    z_t = sigmoid(matmul(X_t, W_z))
    #r_t = sigmoid(matmul(X_t, W_r))
    h_t_hat = tanh(matmul(X_t, W_t))
    h_t = multiply(z_t, h_t_hat)
    output.append(expand_dims(h_t, axis=1))
    h_t_1 = h_t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        z_t = sigmoid(add(matmul(X_t, W_z), matmul(h_t_1, U_z)))
        r_t = sigmoid(add(matmul(X_t, W_r), matmul(h_t_1, U_r)))
        h_t_hat = tanh(add(matmul(X_t, W_t), matmul(multiply(r_t, h_t_1), U_t)))
        h_t = add(multiply(subtract(const_1, z_t), h_t_1), multiply(z_t, h_t_hat))

        output.append(expand_dims(h_t, axis=1))
        h_t_1 = h_t

    y = concatenate(output, axis=1)
    return W_z, U_z, W_r, U_r, W_t, U_t, y

class GRU:
    def __init__(self, lr=0.06):
        self.x = light.functions.placeholder()#(N, 28, 28)
        self.label = light.functions.placeholder()#(N, 10)

        W_z, U_z, W_r, U_r, W_t, U_t, y = gru(self.x, 28, 28, 20, dtype=np.float64)#(N, 28, 20)
        self.W_z = W_z
        self.U_z = U_z
        self.W_r = W_r
        self.U_r = U_r
        self.W_t = W_t
        self.U_t = U_t

        rnn = light.functions.reshape(y, (-1, 560))

        self.out_w = light.functions.variable(get_variable((560, 10), dtype=np.float64))
        self.out_b = light.functions.variable(get_bais((10, ), dtype=np.float64))
        o = light.functions.matmul(rnn, self.out_w)
        o = light.functions.add(o, self.out_b)


        self.objective = softmax_crossentropy(o, self.label)
        self.y = o

        self.compute_forward_train = light.functions.executor(self.objective)
        self.compute_backward_train = light.functions.gradient(self.objective, *[self.W_z, self.U_z, self.W_r, 
        self.U_r, self.W_t, self.U_t, 
        self.out_w, self.out_b])

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
        self.fn(self.W_z, lr*grads[0])
        self.fn(self.U_z, lr*grads[1])    
        self.fn(self.W_r, lr*grads[2])
        self.fn(self.U_r, lr*grads[3])  
        self.fn(self.W_t, lr*grads[4])
        self.fn(self.U_t, lr*grads[5])

        self.fn(self.out_w, lr*grads[6])
        self.fn(self.out_b, lr*grads[7])

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

model = GRU(lr=0.06)
model.fit(x_train, y_train, epochs=20, batch_size=64)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=-1)
acc = pred == y_test
print("accuracy = " + str(acc.mean()))