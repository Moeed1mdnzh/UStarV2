#  Implementation of Inception Score
import numpy as np 

def InceptionScore(X, model, prep, batch_size=32, eps=1E-16):
    processed = X.astype("float32")
    processed = prep(processed)
    yhat = model.predict(processed)
    scores = list()
    for i in range(int(len(X) / batch_size)):
        ix_start, ix_end = l, l+n_split
        p_yx = yhat[ix_start:ix_end]
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std
