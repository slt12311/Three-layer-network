import numpy as np
import scipy.special
class Network():
    """
    定义神经网络
    """
    def __init__(self, input_size, hidden_size, hidden_size_2,output_size, learningrate, activation,reg_strength):
        # 定义输入/隐藏/输出/学习率
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.lr = learningrate
        self.reg_strength = reg_strength
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size_2) * 0.01
        self.b2 = np.zeros((1, self.hidden_size_2))
        self.W3 = np.random.randn(self.hidden_size_2, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))
        # 设置激活函数
        self.activation = activation

    def activate(self,x):
        if self.activation == 'sigmoid':
            return scipy.special.expit(x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)

    def forward(self, X):
        # 前向传播
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activate(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.activate(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def backward(self, X, y, reg_strength):
        def backward_activation(dZ, A, W,activation):
            if activation == "sigmoid":
                return np.dot(dZ, W.T) * (A * (1 - A))
            elif activation == "relu":
                return np.dot(dZ, W.T) * (A > 0)
            elif activation == "tanh":
                return np.dot(dZ, W.T) * (1 - np.power(A, 2))

        # 反向传播
        m = X.shape[0]
        dZ3 = self.A3 - self.one_hot(y, self.output_size)
        dW3 = np.dot(self.A2.T, dZ3) / m + reg_strength / m * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dZ2 = backward_activation(dZ3,self.A2,self.W3,self.activation)
        dW2 = np.dot(self.A1.T, dZ2) / m + reg_strength / m * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = backward_activation(dZ2, self.A1, self.W2, self.activation)
        dW1 = np.dot(X.T, dZ1) / m + reg_strength / m * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 参数更新
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def cross_entropy_loss(self, y_pred, y_true):
        # 交叉熵损失
        m = y_pred.shape[0]
        loss = -np.sum(np.log(y_pred[np.arange(m), y_true])) / m
        return loss

    def softmax(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        m = y.shape[0]
        one_hot_y = np.zeros((m, num_classes))
        one_hot_y[np.arange(m), y] = 1
        return one_hot_y


