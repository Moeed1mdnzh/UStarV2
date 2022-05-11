#  Implementation of Root Mean Squared Error

from numpy import sqrt, mean

def rmse(y, y_pred):
    return sqrt(mean((y-y_pred)**2))