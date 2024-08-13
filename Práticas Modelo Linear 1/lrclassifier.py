from linearregressionp1 import LinearRegression
import numpy as np 

class LRClassifier():
    def execute(self, _X, _y):
      lr = LinearRegression()
      lr.fit_lr(_X, _y)
      self.w = lr.getW_lr()
        
    def predict(self, X):
       return [np.sign(np.dot(self.w, xn)) for xn in X]

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX)/ self.w[2]