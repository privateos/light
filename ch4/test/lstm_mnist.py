import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/../..')
import ch4 as light
import ch4.backend as np

from ch4.functions import multiply, mean, sum, negative, max, subtract, exp, log
from ch4.functions import variable, matmul, add, getitem, tanh, sigmoid, expand_dims, concatenate, multiply

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
C_t_hat = tanh(X_t@W_c + h_t_1@U_c + b_c)
i_t = sigmoid(X_t@W_i + h_t_1@U_i + b_i)
f_t = sigmoid(X_t@W_f + h_t_1@U_f + b_f)
o_t = sigmoid(X_t@W_o + h_t_1@U_o + b_o)
C_t = f_t*C_t_1 + i_t*C_t_hat
"""
def lstm(x, time_step, input_features, output_features, dtype=np.float32, init=uniform):
    output = []
    bias = zeros
    w_shape = (input_features, output_features)
    u_shape = (output_features, output_features)
    b_shape = (output_features, )
    ######C_t_hat##############
    W_c = variable(init(w_shape, dtype=dtype))
    U_c = variable(init(u_shape, dtype=dtype))
    b_c = variable(bias(b_shape, dtype=dtype))

    ######i_t#################
    W_i = variable(init(w_shape, dtype=dtype))
    U_i = variable(init(u_shape, dtype=dtype))
    b_i = variable(bias(b_shape, dtype=dtype))

    #####f_t##################
    W_f = variable(init(w_shape, dtype=dtype))
    U_f = variable(init(u_shape, dtype=dtype))
    b_f = variable(bias(b_shape, dtype=dtype))

    ####o_t##################
    W_o = variable(init(w_shape, dtype=dtype))
    U_o = variable(init(u_shape, dtype=dtype))
    b_o = variable(bias(b_shape, dtype=dtype))

    X_t = getitem(x, (slice(None, None, None), 0, slice(None, None, None)))
    C_t_hat = tanh(add(matmul(X_t, W_c), b_c))
    i_t = sigmoid(add(matmul(X_t, W_i), b_i))
    #f_t = sigmoid(add(matmul(X_t, W_f), b_f))
    o_t = sigmoid(add(matmul(X_t, W_o), b_o))
    C_t = multiply(i_t, C_t_hat)
    h_t = multiply(o_t, tanh(C_t))
    output.append(expand_dims(h_t, axis=1))

    C_t_1 = C_t
    h_t_1 = h_t
    for i in range(1, time_step):
        X_t = getitem(x, (slice(None, None, None), i, slice(None, None, None)))
        C_t_hat = tanh(add(matmul(X_t, W_c), add(matmul(h_t_1, U_c), b_c)))
        i_t = sigmoid(add(matmul(X_t, W_i), add(matmul(h_t_1, U_i), b_i)))
        f_t = sigmoid(add(matmul(X_t, W_f), add(matmul(h_t_1, U_f), b_f)))
        o_t = sigmoid(add(matmul(X_t, W_o), add(matmul(h_t_1, U_o), b_o)))
        C_t = add(multiply(f_t, C_t_1), multiply(i_t, C_t_hat))
        h_t = multiply(o_t, tanh(C_t))

        C_t_1 = C_t
        h_t_1 = h_t
        output.append(expand_dims(h_t, axis=1))
    
    y = concatenate(output, axis=1)
    return W_c, U_c, b_c, W_i, U_i, b_i, W_f, U_f, b_f, W_o, U_o, b_o, y

class LSTM:
    def __init__(self, lr=0.06):
        self.x = light.functions.placeholder()#(N, 28, 28)
        self.label = light.functions.placeholder()#(N, 10)

        W_c, U_c, b_c, W_i, U_i, b_i, W_f, U_f, b_f, W_o, U_o, b_o, y = lstm(self.x, 28, 28, 20, dtype=np.float64)#(N, 28, 20)
        self.W_c = W_c
        self.U_c = U_c
        self.b_c = b_c
        self.W_i = W_i
        self.U_i = U_i
        self.b_i = b_i
        self.W_f = W_f
        self.U_f = U_f
        self.b_f = b_f
        self.W_o = W_o
        self.U_o = U_o
        self.b_o = b_o

        rnn = light.functions.reshape(y, (-1, 560))

        self.out_w = light.functions.variable(get_variable((560, 10), dtype=np.float64))
        self.out_b = light.functions.variable(get_bais((10, ), dtype=np.float64))
        o = light.functions.matmul(rnn, self.out_w)
        o = light.functions.add(o, self.out_b)


        self.objective = softmax_crossentropy(o, self.label)
        self.y = o

        self.compute_forward_train = light.functions.executor(self.objective)
        self.compute_backward_train = light.functions.gradient(self.objective, *[self.W_c, self.U_c, self.b_c, 
        self.W_i, self.U_i, self.b_i, 
        self.W_f, self.U_f, self.b_f, 
        self.W_o, self.U_o, self.b_o,
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
        self.fn(self.W_c, lr*grads[0])
        self.fn(self.U_c, lr*grads[1])    
        self.fn(self.b_c, lr*grads[2])
        self.fn(self.W_i, lr*grads[3])  
        self.fn(self.U_i, lr*grads[4])
        self.fn(self.b_i, lr*grads[5])
        self.fn(self.W_f, lr*grads[6])
        self.fn(self.U_f, lr*grads[7])
        self.fn(self.b_f, lr*grads[8])
        self.fn(self.W_o, lr*grads[9])
        self.fn(self.U_o, lr*grads[10])
        self.fn(self.b_o, lr*grads[11])

        self.fn(self.out_w, lr*grads[12])
        self.fn(self.out_b, lr*grads[13])

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

model = LSTM(lr=0.06)
model.fit(x_train, y_train, epochs=20, batch_size=64)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=-1)
acc = pred == y_test
print("accuracy = " + str(acc.mean()))