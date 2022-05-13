import tensorflow as tf
from itertools import islice


def inference(model, X, y, metric, inception_model=None, prep=None, return_res=True):
    preds = model.predict(X)
    if metric in ["fid", "is"]:
        denormalized = np.uint8(((preds+1.0)/2.0)*255.0)
        score, std = metric(denormalized, inception_model, prep)
    else:
        score = metric(y, preds)
    if return_res:
        return preds, score
    return score

def rank_models(stats, n_epochs):
    stats = {k: v for k, v in sorted(stats.items(), key=lambda x: x[1])}
    ranks = dict(islice(stats.items(), n_epochs if len(stats) < 5 else 5)) 
    return ranks

