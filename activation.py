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
        #print(x)
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


class SoftMax(Function): # SoftMax layer dont have weights
    """
    Class for Softmax function
    """

    def apply(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    
    def derivative(x): 
        # generates J_S_Z
        J_s_z = np.ones((x.shape[1], x.shape[0], x.shape[0]))
        s = SoftMax.apply(x)
        for h in range(x.shape[1]):
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    if j == i:
                        J_s_z[h,i,j] = s[j,h]-s[j,h]**2
                    else: # se om det er riktig indeksert om noe går galt
                        J_s_z[h,i,j] = -s[j,h]*s[i,h]
        return J_s_z


class MSE(Function):
    """
    Class for Mean Squared error
    """

    def apply(pred, target):
        return (1/pred.shape[1])*np.sum((pred-target)**2, axis=1).reshape(-1,pred.shape[0])

    def derivative(pred, target): #with respect to pred
        return ((2/pred.shape[0])*(pred-target)).reshape(pred.shape[0],-1)

class MSE_L2(MSE):

    def apply(pred, target, weight_reg):
        return (1/pred.shape[1])*np.sum((pred-target)**2, axis=1).reshape(-1,pred.shape[0]) + weight_reg
    
    def derivative(pred, target):
        return super().derivative(target)


class CrossEntropy(Function): 
    """
    Class for Cross Entropy error
    """
    def apply(pred, target):
        return - np.sum(target*np.log2(pred), axis=0)

    def derivative(pred, target):
        return (-target*1/(pred*np.log(2))).reshape(pred.shape[0],-1)



if __name__ == "__main__":
    x = np.zeros((2,4))
    y = np.zeros((2,4))
    
    x[0,0] = 0
    x[1,0] = 0
    y[0,0] = 0
    y[1,0] = 0
    

    x[0,1] = 1
    x[1,1] = 0
    y[0,1] = 0
    y[1,1] = 1

    x[0,2] = 0
    x[1,2] = 1
    y[0,2] = 0
    y[1,2] = 1

    x[0,3] = 1
    x[1,3] = 1
    y[0,3] = 1
    y[1,3] = 0

    print(MSE.derivative(x,y).T[:,np.newaxis,:])
   