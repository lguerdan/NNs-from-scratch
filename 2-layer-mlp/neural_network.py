import numpy as np
from numpy import genfromtxt

class NeuralNetwork:
    

    def __init__(self, input_dim, hidden_dim, output_dim, alpha, beta):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        self.SSE = 0
        self.SSE_epochs = []
        
        
    #Initialize network parameters    
    def init_weights(self, W1_file, W2_file, b1_file, b2_file):
        '''hard code weight initialization from given files'''
        self.W1 = genfromtxt(W1_file, delimiter=',' , dtype=np.float128)
        self.W2 = genfromtxt(W2_file, delimiter=',', dtype=np.float128)
        
        self.W1_prev_update = np.zeros(self.W1.shape, dtype=np.float128)
        self.W2_prev_update = np.zeros(self.W2.shape, dtype=np.float128)
        
        self.b1 = genfromtxt(b1_file, delimiter=',')
        self.b2 = genfromtxt(b2_file, delimiter=',')
        
        self.bias_update_1_prev = np.zeros(self.b1.shape, dtype=np.float128)
        self.bias_update_2_prev = np.zeros(self.b2.shape, dtype=np.float128)
        
        self.y1 = np.zeros(self.hidden_dim)
        self.y2 = np.zeros(self.output_dim)
    
    def init_weights_rand(self):
        self.W1 = np.random.rand(self.hidden_dim, self.input_dim)
        self.W2 = np.random.rand(self.output_dim, self.hidden_dim)
        
        self.W1_prev_update = np.zeros(self.W1.shape)
        self.W2_prev_update = np.zeros(self.W2.shape)
        
        self.b1 = np.random.rand(self.hidden_dim)
        self.b2 = np.random.rand(self.output_dim)
        
        self.bias_update_1_prev = np.zeros(self.b1.shape)
        self.bias_update_2_prev = np.zeros(self.b2.shape)
        
        self.y1 = np.zeros(self.hidden_dim)
        self.y2 = np.zeros(self.output_dim)
        
        self.SSE = 0
        self.SSE_epochs = []
        
    
    #Train over training dataset over specified epochs
    def train(self, epochs, features, lables, delta):
        
        for epoch in range(epochs):
            for sample in range(features.shape[0]):
                self.forward_pass(features[sample,:], lables[sample,:])
                self.backward_pass(features[sample,:])
            
            
            p = np.random.permutation(len(features))
            features, lables = features[p], lables[p]
            self.SSE = self.SSE /2 #SSE For this epoch
            self.SSE_epochs.append(self.SSE)
            if delta > 0 and len(self.SSE_epochs) > 2 and np.abs(np.diff(self.SSE_epochs[-2:])[0]) < delta:
                return
            self.SSE = 0
    
    def evaluate(self, features):
        
        output_lables = np.zeros([features.shape[0],self.output_dim])
        self.SSE = 0
        for sample in range(features.shape[0]):
            output_lables[sample,:] = np.transpose(self.forward_pass_evaluate(features[sample,:]))
        return output_lables
    
    #Compute a forwared pass on the data
    def forward_pass_evaluate(self, features):
        
        self.y1  = self.W1.dot(features) + self.b1
        self.y1 = self.sigmoid(self.y1)
        
        self.y2 = self.W2.dot(self.y1) + self.b2
        self.y2 = self.sigmoid(self.y2)
        
        self.SSE += np.sum(np.square(self.error))
        return self.y2
        
            
        
    #Compute a forwared pass on the data
    def forward_pass(self, features, target):
        
        self.y1  = self.W1.dot(features) + self.b1
        self.y1 = self.sigmoid(self.y1)
        
        self.y2 = self.W2.dot(self.y1) + self.b2
        self.y2 = self.sigmoid(self.y2)
        
        self.error = target - self.y2
        self.SSE += np.sum(np.square(self.error))
    
    #Compute a backward pass on the data
    def backward_pass(self, features):

        d2 = np.multiply(self.error, self.sigmoid_p(self.y2))
        grad_update2 = np.multiply(self.alpha, np.outer(d2, self.y1))
        momentum2 = np.multiply(self.beta, self.W2_prev_update)
        
        d1 = np.multiply(self.sigmoid_p(self.y1), np.transpose(self.W2).dot(d2))
        grad_update1 = np.multiply(self.alpha, np.outer(d1,features))
        momentum1 = np.multiply(self.beta, self.W1_prev_update)
        
        grad_update_bias2 = np.multiply(self.alpha, d2);
        momentum_bais2 = np.multiply(self.beta, self.bias_update_2_prev);
        bias_update_2 = grad_update_bias2 + momentum_bais2
        self.b2 = self.b2 + bias_update_2
        self.bias_update_2_prev = bias_update_2
        
        
        grad_update_bias1 = np.multiply(self.alpha, d1);
        momentum_bais1 = np.multiply(self.beta, self.bias_update_1_prev);
        bias_update_1 = grad_update_bias1 + momentum_bais1
        self.b1 = self.b1 + bias_update_1
        self.bias_update_1_prev = bias_update_1
            
        self.W2 = self.W2 + momentum2 + grad_update2
        self.W2_prev_update = momentum2 + grad_update2
                
        self.W1 = self.W1 + momentum1 + grad_update1
        self.W1_prev_update =  momentum1 + grad_update1
        
        
    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(- array))
    
    @staticmethod
    def sigmoid_p(array):
        return np.multiply(array, (1 - array))





