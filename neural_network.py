#��������������������
#1.��ʼ���������������趨����ڵ㡢���ز�ڵ㡢�����ڵ������
#2.��ѵ��������������ѧϰ����ѵ�����������Ż�Ȩ��
#3.  ��ѯ�������������������룬������ڵ������

#��������Ķ���
import numpy 
import scipy.special

class Neural_Network(object):
    
    # ��ʼ��������
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #��������ڵ㡢���ؽڵ㡢����ڵ�ĸ���
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        #���ó�ʼȨ��(����������ֵ����׼��������С)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #����ѧϰ��
        self.lr = learning_rate

        #����sigmoid(x)Ϊ�����
        self.activation_function = lambda x: scipy.special.expit(x)

    #ѵ��
    def train(self, input_list, targets_list):
        #ת�������б�Ϊ����
        inputs = numpy.array(input_list, ndmin=2)
        #ת����ֵ�б�Ϊ����
        targets = numpy.array(targets_list, ndmin=2)
        #�������ز�����
        hidden_inputs = numpy.dot(self.wih, inputs)
        #�������ز����
        hidden_outputs = self.activation_function(hidden_inputs)
        #�������������
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #����������ս��
        final_outputs = self.activation_function(final_inputs)
        #�������
        output_errors = targets - final_outputs
        #�������ز��������
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #����Ȩ��--���ز㡢�����
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        #����Ȩ��--����㡢���ز�
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs) )
        

    #��ѯ
    def query(self, input_list):
        #���������б�
        inputs = numpy.array(input_list, ndmin=2).T
        #�������ز�����
        hidden_inputs = numpy.dot(self.wih, inputs)
        #�������ز����
        hidden_outputs = self.activation_function(hidden_inputs)
        #�������������
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #����������ս��
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

