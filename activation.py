import numpy as np

class Function:
    """
    Abstract parent class for function
    """

    def apply(x):
        pass
    
    def derivative(x):
        pass


class Sigmoid(Function):
    """
    Class for the sigmoid function
    """

    def apply(x):
        return 1/(1+np.e**(-x))

    def derivative(x):
        return Sigmoid.apply(x)*(1-Sigmoid.apply(x))


class Relu(Function):
    """
    Class for the relu function
    """

    def apply(x):
        return x * (x > 0)
    
    def derivative(x):
        return (x > 0)*1


class Tanh(Function):
    """
    Class for the tanh function
    """

    def apply(x):
        return ((np.e**x) - (np.e**(-x)))/((np.e**x) + (np.e**(-x)))

    def derivative(x):
        return 1 - Tanh.apply(x)**2
    

class Linear(Function):
    """
    Class for linear function
    """

    def apply(x):
        return x

    def derivative(x):
        return np.ones(x.shape)

class MSE(Function):
    """
    Class for Mean Squared error
    """

    def apply(pred, target):
        return (1/pred.shape[1])*np.sum((pred-target)**2, axis=1).reshape(-1,1)

    def derivative(pred, target): #with respect to pred
        return ((2/pred.shape[0])*(pred-target)).reshape(1,-1)


class SoftMax(Function):

    def apply(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    
    def derivative(x):
        return super().derivative()




if __name__ == "__main__":
    print(Sigmoid.apply(0))
    print(Sigmoid.derivative(np.array([0.01578004, 0.07877845, 0.00127282])))
    print(SoftMax.apply(np.array([[3], [1], [0.2]])))
    print(np.array([[3], [1], [0.2]]))
    