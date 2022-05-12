#  Implementation of Mean Squared Error

from numpy import mean

def MeanSquaredError(y, y_pred):
    return mean((y-y_pred)**2)