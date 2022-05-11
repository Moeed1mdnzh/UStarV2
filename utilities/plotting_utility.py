import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def preview(X, y, batch_size=4):
    indices = np.random.randint(len(X), size=batch_size)
    X_batch = X[indices]
    y_batch = y[indices]
    images = []
    for Ximg, Yimg in zip(X_batch, y_batch):
        images.append(Ximg)
        images.append(Yimg)
    fig, axs = plt.subplots(batch_size, 2, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img)
        ax.axis("off")
    return fig

def plot_training(images, n_cols, n_rows):
    fig = plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        if index == n_cols * n_rows:
            break
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow((image + 1.0)/2.0)
        plt.axis("off")
    return fig