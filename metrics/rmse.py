#  Implementation of Root Mean Squared Error
from numpy import sqrt, mean

def RootMeanSquaredError(y, y_pred):
    return sqrt(mean((y-y_pred)**2))