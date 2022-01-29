
import matplotlib.pyplot as plt
import numpy as np
import math
from activation import CrossEntropy, Sigmoid
from generator import Generator
from layer import Layer, SoftMaxLayer

from network import NeuralNetwork
from config_data import ConfigData
"""
# try to use the data images and the neural net together

from config_network import ConfigNetwork


cd = ConfigData()
cn = ConfigNetwork()

nn = cn.config_neural_network()
data = cd.generate_data()
print("generated")

x = data[0][0]

x = data[0][0].reshape(x.shape[1], x.shape[0])
#print(x)

y = data[0][1]


#print(y)

#batch = 2
#number = math.ceil(y.shape[1]/batch)
#print(number)
#y_batch = np.array_split(y, number, axis=1)
#for i in range(number):
#    print(y_batch[i])


nn.train(x, y, 500, batch=10)


x_test = data[2][0]
y_test = data[2][1]

x_test = x_test.reshape(x_test.shape[1], x_test.shape[0])
print(x_test)
print(y_test)


pred = nn.forward_pass(x_test)
print(pred[:,0])
print(y_test[:,0])

print(pred[:,1])
print(y_test[:,1])

print(pred[:,2])
print(y_test[:,2])

print(pred[:,3])
print(y_test[:,3])

print(nn.loss_graph[0])
print(nn.loss_graph[-1])


# plotting graph
plt.plot(nn.batch_graph, nn.loss_graph)
plt.show()

#print(y)
"""
g = Generator()

nn = NeuralNetwork(loss_function=(CrossEntropy.apply, CrossEntropy.derivative))

nn.add_hidden_layer(Layer(100, 400, (Sigmoid.apply, Sigmoid.derivative), lrate=0.1, wr=[-0.5,0.5], br=[-0.5,0.5]))
nn.add_hidden_layer(Layer(4,100, (Sigmoid.apply, Sigmoid.derivative), lrate=0.1, wr=[-0.5,0.5], br=[-0.5,0.5]))
nn.set_softmax_layer(SoftMaxLayer())

data = g.generate(number=500, size=20, split=(0.7,0.2,0.1), noise=0.05, object_height_range=(10,20), object_width_range=(10,20), centerd=True, flattend=True)

#dette er sånn man går fra (m,n,1) til (m,n)
x = np.squeeze(data[0][0]).T

y = data[0][1]

nn.train(x, y, 100, batch=20)

x_test = data[2][0]
y_test = data[2][1]

x_test = np.squeeze(x_test).T

pred = nn.forward_pass(x_test)
print(pred[:,0])
print(y_test[:,0])

print(pred[:,1])
print(y_test[:,1])

print(pred[:,2])
print(y_test[:,2])

print(pred[:,3])
print(y_test[:,3])

print(nn.loss_graph[0])
print(nn.loss_graph[-1])

plt.plot(nn.batch_graph, nn.loss_graph)
plt.show()