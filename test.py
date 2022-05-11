import tensorflow as tf
from itertools import islice


def inference(model, X, y, return_res=True):
    preds = model.predict(X)
    mse = tf.keras.metrics.MeanSquaredError()(preds, y)
    if return_res:
        return preds, mse
    return mse

def rank_models(stats, n_epochs):
    stats = {k: v for k, v in sorted(stats.items(), key=lambda x: x[1])}
    ranks = dict(islice(stats.items(), n_epochs if len(stats) < 5 else 5)) 
    return ranks

