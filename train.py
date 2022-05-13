import os
import cv2 
import json
import logging
import argparse
import numpy as np 
from test import *
from metrics import *
import tensorflow as tf
from UNet.configs import *
from InceptionV3.configs import *
from matplotlib import pyplot as plt
from utilities.augmentation import *
from imutils.paths import list_images
from utilities.plotting_utility import *
from UNet.generator.generator import Generator
from InceptionV3.inceptionv3 import InceptionV3
from sklearn.model_selection import train_test_split
from UNet.discriminator.discriminator import Discriminator 

tf.get_logger().setLevel(logging.ERROR)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--visualize", default=True, type=bool,
                help="To wether save the visualizations or not")
ap.add_argument("-e", "--evaluate", default=False, type=bool,
                help="To wether evaluate the model or not")
ap.add_argument("-m", "--metric", default="is", type=str, choices=["mse", "mae", 
                                                                   "rmse", "is",
                                                                   "fid"],
                help="The metric used for evaluation after training")
ap.add_argument("-r", "--rank", required=True, type=str, choices=["mse", "mae", 
                                                                   "rmse", "is",
                                                                   "fid"],
                help="The metric used for the ranking system")
ap.add_argument("-s", "--save", default=True, type=bool,
                help="To wether save the best model or not")
args = vars(ap.parse_args())

X_images = []
X_targets = []
paths = list_images(os.sep.join(["data", "images"]))
for path in paths:
    image = cv2.imread(path)
    label = cv2.imread(path.replace("images", "labels"))
    X_images.append(cv2.resize(image, (128, 128)))
    X_targets.append(cv2.resize(label, (128, 128)))

X_images = np.asarray(X_images)
X_targets = np.asarray(X_targets)
print(f"Images shape: {X_images.shape}\nTargets shape: {X_targets.shape}")


if args["visualize"]:
    fig = preview(X_images, X_targets)
    plt.savefig("data_preview.png", dpi=fig.dpi)
    
X_images_train, X_images_test, X_targets_train, X_targets_test = train_test_split(X_images, X_targets,
                                        test_size=0.25, random_state=42)

X_images_test = (X_images_test.astype("float32")-127.5)/127.5
X_targets_test = (X_targets_test.astype("float32")-127.5)/127.5

print(f"Train images shape: {X_images_train.shape}\nTrain targets shape: {X_targets_train.shape}\n")
print(f"Test images shape: {X_images_test.shape}\nTest targets shape: {X_targets_test.shape}")

augmented_X_images = []
augmented_X_targets = []

for i, Ximg in  enumerate(X_images_train):
    print(f"[INFO]: Augmenting image number {i+1}")
    augmented_X_images.append(Ximg)

    mirrored_X = mirror(Ximg)

    light_X = lighting(Ximg, 0.2)

    rotated_X = rotate(Ximg)

    zoomed_outX = zoom_out(Ximg)

    transferred_X = transfer(Ximg, 0.4) 

    for Xset in [mirrored_X, light_X, rotated_X, zoomed_outX, transferred_X]:
        for Ximg in Xset:
            augmented_X_images.append(Ximg)

#  Seperating images and targets sets loops to avoid session crash


for i, Ximg in  enumerate(X_targets_train):
    print(f"[INFO]: Augmenting target number {i+1}")
    augmented_X_targets.append(Ximg)

    mirrored_X = mirror(Ximg)

    light_X = lighting(Ximg, 0.2)

    rotated_X = rotate(Ximg)

    zoomed_outX = zoom_out(Ximg)

    transferred_X = transfer(Ximg, 0.4) 

    for Xset in [mirrored_X, light_X, rotated_X, zoomed_outX, transferred_X]:
        for Ximg in Xset:
            augmented_X_targets.append(Ximg)
     
special_metrics = ["fid", "is"]  
inception_model = None   
if any([args[arg] in special_metric for arg in ["metric", "rank"]]):    
    inception_X, inception_y = [], []         
    for image in  X_targets:
        inception_X.append(image)
        inception_y.append(0)
        
        mirrored_X = mirror(image)

        rotated_X = rotate(image)

        zoomed_outX = zoom_out(image)

        transferred_X = transfer(image, 0.4) 

        for i, Xset in enumerate([mirrored_X, rotated_X, zoomed_outX, transferred_X]):
            for Ximg in Xset:
                inception_X.append(Ximg)
                inception_y.append(i+1)
                
    inception_X = np.array(inception_X).astype("float32")
    inception_y = tf.keras.utils.to_categorical(y, 5)
    incX_train, incX_test, incy_train, incy_test = train_test_split(inception_X, 
                                                                    inception_y,
                                                                    test_size=0.25,
                                                                    random_state=42,
                                                                    shuffle=True)
    print(f"Inception train images shape: {X_train.shape}\nInception train labels shape: {y_train.shape}\n")
    print(f"Inception Test images shape: {X_test.shape}\nInception Test labels shape: {y_test.shape}")
    incX_train, incX_test = X_train / 255.0, X_test / 255.0
    inception_model, inception_prep = InceptionV3(NUM_CLASSES, FREEZE_LAYERS, INCEPTION_SHAPE).build_inception_model()
    for epoch in range(INCEPTION_EPOCHS):
        epoch_widgets = inception_widgets.copy()
        epoch_widgets[0] = epoch_widgets[0] % (str(epoch+1), str(INCEPTION_EPOCHS))
        pbar = progressbar.ProgressBar(maxval=len(incX_train) // INCEPTION_BATCH_SIZE, 
                                       widgets=epoch_widgets).start()
        for i, j in enumerate(range(0, len(incX_train), INCEPTION_BATCH_SIZE)):
            X_batch, y_batch = incX_train[j: j+INCEPTION_BATCH_SIZE], incy_train[j: j+INCEPTION_BATCH_SIZE]
            with tf.GradientTape() as tape:
                y_pred = inception_model(X_batch, training=True)
                loss = inception_loss(y_batch, y_pred)
            gradients = tape.gradient(loss, inception_model.trainable_weights)
            inception_opt.apply_gradients(zip(gradients, inception_model.trainable_weights))
            y_pred = inception_model.predict(X_batch)
            acc = acc_fn(np.argmax(y_batch, axis=1), np.argmax(y_pred, axis=1))
            loss = inception_loss2(y_batch, y_pred)
            indices = np.random.randint(0, len(incX_test), INCEPTION_TEST_SAMPLE)
            y_pred = inception_model.predict(incX_test[indices])
            val_acc = acc_fn(np.argmax(incy_test[indices], axis=1), np.argmax(y_pred, axis=1))
            val_loss = val_inception_loss(incy_test[indices], y_pred)
            pbar.update(i, inception_loss=loss, inception_accuracy=acc,
                        val_inception_loss=val_loss, val_inception_accuracy=val_acc)
        pbar.finish()
    inception_model.save(os.sep.join(["InceptionV3", "inceptionv3.h5"]))

augmented_X_images, augmented_X_targets = np.array(augmented_X_images), np.array(augmented_X_targets)
indices = np.arange(augmented_X_images.shape[0])
np.random.shuffle(indices)
augmented_X_images = augmented_X_images[indices]
augmented_X_targets = augmented_X_targets[indices]

if args["visualize"]:
    fig = preview(augmented_X_images, augmented_X_targets)
    plt.savefig("augmented_data_preview.png", dpi=fig.dpi)

print(f"Augmented images shape: {augmented_X_images.shape}\nAugmented targets shape: {augmented_X_targets.shape}")

augmented_X_images = (augmented_X_images.astype("float32")-127.5)/127.5
augmented_X_targets = (augmented_X_targets.astype("float32")-127.5)/127.5

discriminator = Discriminator(DISC_SHAPE, tf.keras.initializers.RandomNormal(stddev=0.02)).build_model()
generator = Generator(GEN_SHAPE, tf.keras.initializers.RandomNormal(stddev=0.02)).build_model()

discriminator.trainable = False

for epoch in range(N_EPOCHS):
    epoch_widgets = widgets.copy()
    epoch_widgets[0] = epoch_widgets[0] % (str(epoch+1), str(N_EPOCHS))
    pbar = progressbar.ProgressBar(maxval=len(augmented_X_images) // BATCH_SIZE, 
                                   widgets=epoch_widgets).start()
    for i, j in enumerate(range(0, len(augmented_X_images), BATCH_SIZE)):
        batch_images = augmented_X_images[j: j+BATCH_SIZE]
        batch_targets = augmented_X_targets[j: j+BATCH_SIZE]

        generated_targets = generator(batch_images, training=False)
        y1 = tf.concat((tf.zeros((len(batch_images), 1)), tf.ones((len(batch_images), 1))), axis=0)
        disc_samples = tf.concat((generated_targets, batch_targets), axis=0)
        disc_samples = tf.concat((disc_samples, tf.concat((batch_images, batch_images), axis=0)), axis=3)

        discriminator.trainable = True
        with tf.GradientTape() as tape:
            y_pred = discriminator(disc_samples, training=True)
            d_loss = disc_loss(y1, y_pred) * 0.5
        gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        disc_opt.apply_gradients(zip(gradients, discriminator.trainable_variables))

        discriminator.trainable = False
        y2 = tf.ones((len(batch_images), 1))
        with tf.GradientTape() as tape:
            fake_images = generator(batch_images, training=True)
            fake_images = tf.concat((fake_images, batch_images), axis=3)
            y_pred = discriminator(fake_images, training=False)
            g_loss1 = tf.expand_dims(gen_loss1(y2, y_pred), axis=0)
            g_loss2 = tf.expand_dims(gen_loss2(y2, y_pred), axis=0)
            g_loss = tf.concat((g_loss1, g_loss2), axis=0)
            #  Apply weighted sum
            g_loss = tf.multiply(g_loss, tf.constant([1., 100.]))
            g_loss = tf.add(g_loss[0], g_loss[1])
        gradients = tape.gradient(g_loss, generator.trainable_variables)
        gen_opt.apply_gradients(zip(gradients, generator.trainable_variables))
        pbar.update(i, g_loss=g_loss, d_loss=d_loss)

    generator.save(os.sep.join(["UNet", "models", f"generator_{epoch+1}.h5"]))
    if args["visualize"]:
        fig = plot_training(generator(augmented_X_images[:N_ROWS * N_COLS], training=False), N_COLS, N_ROWS)
        plt.savefig(os.sep.join(["UNet", "visualizations", f"viz_{epoch+1}.png"]), dpi=fig.dpi)
    pbar.finish()

stats = {}
metric_names = {"mse": mse.MeanSquaredError,
                "mae": mae.MeanAbsoluteError,
                "rmse": rmse.RootMeanSquaredError,
                "is": IS.InceptionScore ,
                "fid": fid.FrechetInceptionDistance}
for i in range(N_EPOCHS):
    model = tf.keras.models.load_model(os.sep.join(["UNet", "models", f"generator_{i+1}.h5"]))
    score = inference(model, X_images_test, X_targets_test,
                      metric_names[args["rank"]], inception_model, inception_prep, return_res=False)
    stats[str(i+1)] = float(score.numpy())
ranks = rank_models(stats, N_EPOCHS)

if args["save"]:
    best = min(ranks.items())[0]
    path = os.sep.join(["UNet", "best_model"])
    os.system(f"mkdir {path}")
    model = tf.keras.models.load_model(os.sep.join(["UNet", "models", f"generator_{best}.h5"]))
    model.save(os.sep.join(["UNet", "best_model", "best_model.h5"]))

if args["evaluate"]:
    index = np.random.randint(0, X_images_test.shape[0])
    _input, target = X_images_test[index], X_targets_test[index]
    fig = plt.figure(figsize=(2, 6))
    plt.subplot(6, 2, 1)
    plt.imshow((_input + 1.0)/2.0)
    plt.axis("off")
    plt.title("input")

    plt.subplot(6, 2, 2)
    plt.imshow((target + 1.0)/2.0)
    plt.axis("off")
    plt.title("target")
    k = 3
    for i, (model_id, score) in enumerate(ranks.items()):
        plt.subplot(6, 2, i + k)
        plt.imshow((_input + 1.0)/2.0)
        plt.axis("off")
        plt.subplot(6, 2, i + k + 1)
        model = tf.keras.models.load_model(os.sep.join(["UNet", "models", f"generator_{model_id}.h5"]))
        pred, score = inference(model, np.expand_dims(_input, 0), target,
                                metric_names[args["rank"]], inception_model, inception_prep)
        plt.imshow((pred.squeeze()+1.0)/2.0)
        plt.axis("off")
        k += 1
    metric_name = args["metric"]
    with open(f"{name}_ranks.json", "w", encoding="utf-8") as f:
        json.dump(ranks, f, ensure_ascii=False, indent=4)
    plt.savefig("evaluation_result.png", dpi=fig.dpi)
