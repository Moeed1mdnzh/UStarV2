#  Implementation of Mean Absolute Error

from numpy import mean, absolute 

def MeanAbsoluteError(y, y_pred):
    return mean(absolute(y-y_pred))