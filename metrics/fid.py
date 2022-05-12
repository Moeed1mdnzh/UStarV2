#  Implementation of Frechet Inception Distance
import numpy as np


def FrechetInceptionDistance(X, model, prep, batch_size, eps=1E-16):
    scores = list()
    l = 0
    for i in range(int(len(images) / batch_size)):
        ix_start, ix_end = l, l+batch_size
        subset = X[ix_start: ix_end]
        subset = subset.astype("float32")
        subset = prep(subset)
        p_yx = model.predict(subset)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
        l += batch_size
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std