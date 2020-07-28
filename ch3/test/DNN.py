import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/../..')
import ch3 as light
import ch3.backend as np


from ch3.functions import multiply, mean, sum, negative, max, subtract, exp, log

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

def softplus(x):
    ex = light.functions.exp(x)
    const = light.functions.constant(1.0)
    return light.functions.log(light.functions.add(const, ex))

def get_variable(shape, dtype):
    return np.random.uniform(-0.5, 0.5, shape).astype(dtype)
def get_bais(shape, dtype):
    return np.zeros(shape, dtype=dtype)


class DNN:
    def __init__(self, lr=0.001):
        self.x = light.functions.placeholder()#(N, 784)
        self.label = light.functions.placeholder()#(N, 10)

        self.hidden1_w = light.functions.variable(get_variable((784, 256), dtype=np.float64))
        self.hidden1_b = light.functions.variable(get_bais((256, ), dtype=np.float64))

        self.hidden2_w = light.functions.variable(get_variable((256, 128), dtype=np.float64))
        self.hidden2_b = light.functions.variable(get_bais((128, ), dtype=np.float64))

        self.output_w = light.functions.variable(get_variable((128, 10), dtype=np.float64))
        self.output_b = light.functions.variable(get_bais((10, ), dtype=np.float64))


        h1 = light.functions.matmul(self.x, self.hidden1_w)
        h1 = light.functions.add(h1, self.hidden1_b)
        h1 = light.functions.tanh(h1)

        h2 = light.functions.matmul(h1, self.hidden2_w)
        h2 = light.functions.add(h2, self.hidden2_b)
        h2 = softplus(h2)

        o = light.functions.matmul(h2, self.output_w)
        o = light.functions.add(o, self.output_b)
        self.objective = softmax_crossentropy(o, self.label)
        self.y = o

        self.compute_forward_train = light.functions.executor(self.objective)
        self.compute_backward_train = light.functions.gradient(self.objective, *[self.hidden1_w, self.hidden1_b, self.hidden2_w, self.hidden2_b, self.output_w, self.output_b])

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
        self.fn(self.hidden1_w, lr*grads[0])
        self.fn(self.hidden1_b, lr*grads[1])    
        self.fn(self.hidden2_w, lr*grads[2])  
        self.fn(self.hidden2_b, lr*grads[3])
        self.fn(self.output_w, lr*grads[4])
        self.fn(self.output_b, lr*grads[5])

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
x = np.reshape(x, (x.shape[0], -1))
x_train, x_test = x[0:60000], x[60000:]
y_train, y_test = y[0:60000], y[60000:]

model = DNN(lr=0.06)
model.fit(x_train, y_train, epochs=20, batch_size=64)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
y_test = np.argmax(y_test, axis=-1)
acc = pred == y_test
print("accuracy = " + str(acc.mean()))