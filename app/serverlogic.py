import pandas as pd
import numpy as np
import plotly.express as plt

class LinearRegression():
    def __init__(self) -> None:
        self.__beta = []

    def __calc_linear(self,X, beta):
        # X --> m * (n+1), beta --> (n+1)*1, output --> m*1 column vector
        return np.matmul(X,beta) # multiply each beta value with the corresponding x feature value

    def fit(self,x_train,y_train,alpha=0.01,iterations=2000):
        # compute beta and store it 
        beta = np.zeros(shape=(x_train.shape[1],1)) # column vector of shape (n+1) * 1 
        self.__beta = self.__gradient_descent(x_train,y_train,beta,alpha,iterations)
        
    def __compute_cost(self,X, y, beta):
        y_hat = self.__calc_linear(X,beta)
        m = len(X)
        J = sum((y_hat-y)**2)
        return (1/(2*m))*J

    def __gradient_descent(self,X, y, beta, alpha, num_iters):
        m = X.shape[0]
        J_storage = np.zeros((num_iters,1))
        for n in range(num_iters):
            deriv = np.matmul(X.T,(self.__calc_linear(X,beta)-y))
            beta = beta - alpha * (1/m) * deriv
            J_storage[n] = self.__compute_cost(X,y,beta)
        return beta

    def predict(self,x_test):
        x_test = np.array(list(map(float,x_test)))
        x_test = np.concatenate((np.ones(shape=(1,1)),x_test.reshape(1,len(x_test))),axis=1)
        assert x_test.shape[1] == self.__beta.shape[0], f"X_test shape: {x_test.shape}, beta_shape: {self.__beta.shape} do not match"
        return self.__calc_linear(x_test,self.__beta)

from app.utility import normalize_vector_dict, train_and_predict
            

# take non-normalized data and make it normalized 
def make_prediction(x_dict):
    m = LinearRegression()
    y_pred = train_and_predict(m,x_dict)
    #x_dict = normalize_vector_dict(x_dict)
    return round(y_pred[0],2)

def make_comparison(current_imports,predicted_imports):
    difference = abs(current_imports-predicted_imports)
    output = ""
    less = False
    if current_imports < predicted_imports:
        # not importing enough
        output += f"You are importing lesser than what this model predicts, by an index value of {difference}"
        less = True
    else:
        output += f"You are importing more than what this model predicts, by an index value of {difference}"
    return output,less

