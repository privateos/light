import sys
import os
sys.path.append(os.path.split(os.path.realpath(__file__))[0]+'/../..')
import ch2 as light
import ch2.backend as np

def get_variable(shape, dtype):
    return np.random.uniform(-0.5, 0.5, shape).astype(dtype)
def get_bais(shape, dtype):
    return np.zeros(shape, dtype=dtype)
def mean_squared_error(y, label):
    loss = light.functions.subtract(y, label)
    loss = light.functions.square(loss)
    loss = light.functions.mean(loss, axis=0)
    loss = light.functions.sum(loss)
    return loss

class LinearRegression:
    def __init__(self, input_dims, output_dims, lr=0.001):

        self.x = light.functions.placeholder()
        self.w = light.functions.variable(get_variable((input_dims, output_dims), dtype=np.float64))
        self.b = light.functions.variable(get_bais((output_dims, ), dtype=np.float64))
        self.y = light.functions.add(light.functions.matmul(self.x, self.w), self.b)

        self.label = light.functions.placeholder()
        self.objective = mean_squared_error(self.y, self.label)
        self.compute_forward_train = light.functions.executor(self.objective)
        self.compute_backward_train = light.functions.gradient(self.objective, *[self.w, self.b])

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
        self.fn(self.w, self.lr*grads[0])
        self.fn(self.b, self.lr*grads[1])        
    
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
    
    def predict(self, x):
        feed_dict = {self.x:x}
        r, = self.compute_forward_test.run(feed_dict)
        return r


from sklearn.datasets import make_classification

x,y = make_classification(n_samples=20000, n_features=2, n_informative=2,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=1, weights=None,
                          flip_y=0.01, class_sep=1.0, hypercube=True,
                          shift=0.0, scale=1.0, shuffle=True, random_state=None)

import matplotlib.pyplot as plt                          
y = y.reshape((y.shape[0], 1))
LR = LinearRegression(2, 1, lr=0.0005)
LR.fit(x, y, 40, batch_size=256)
pred = LR.predict(x)
p = np.where(pred > 0.5, 1, 0)
acc = p == y
print("accuracy="+str(acc.mean()))

plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=1)
plt.show()
p = np.where(pred > 0.5, 1, 0)
plt.scatter(x[:, 0], x[:, 1], c=p[:, 0], s=1)
plt.show()
