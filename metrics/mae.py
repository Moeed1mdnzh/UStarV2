#  Implementation of Mean Absolute Error

from numpy import mean, absolute 

def mae(y, y_pred):
    return mean(absolute(y-y_pred))