#神经网络至少有三个函数
#1.初始化函数————设定输入节点、隐藏层节点、输出层节点的数量
#2.　训练　　————学习给定训练集样本后，优化权重
#3.  查询　　————给定输入，从输出节点给出答案

#神经网络类的定义
import numpy 
import scipy.special

class Neural_Network(object):
    
    # 初始化神经网络
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #设置输入节点、隐藏节点、输出节点的个数
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        #设置初始权重(参数：中心值，标准方差，矩阵大小)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #设置学习率
        self.lr = learning_rate

        #设置sigmoid(x)为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    #训练
    def train(self, input_list, targets_list):
        #转化输入列表为数组
        inputs = numpy.array(input_list, ndmin=2)
        #转化真值列表为数组
        targets = numpy.array(targets_list, ndmin=2)
        #计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算输出最终结果
        final_outputs = self.activation_function(final_inputs)
        #计算误差
        output_errors = targets - final_outputs
        #计算隐藏层误差数组
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #更新权重--隐藏层、输出层
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        #更新权重--输入层、隐藏层
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs) )
        

    #查询
    def query(self, input_list):
        #设置输入列表
        inputs = numpy.array(input_list, ndmin=2).T
        #计算隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算输出最终结果
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

