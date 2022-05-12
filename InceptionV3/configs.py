import progressbar
import tensorflow as tf 
from sklearn.metrics import accuracy_score

NUM_CLASSES = 5

FREEZE_LAYERS = 104

INCEPTION_SHAPE = (128, 128, 3)

inception_opt = tf.keras.optimizers.SGD(learning_rate=0.01,
                                        decay=0.01/40,
                                        momentum=0.9,
                                        nesterov=True)

inception_widgets = ["Epoch %s/%s ", progressbar.Percentage(), " ", progressbar.Bar(),
           " ", progressbar.ETA(), " ", progressbar.Variable("inception_loss"),
           " ", progressbar.Variable("inception_accuracy"),  
           " ", progressbar.Variable("val_inception_loss"),
           " ", progressbar.Variable("val_inception_accuracy")]

inception_loss = tf.keras.losses.CategoricalCrossentropy()
inception_loss2 = tf.keras.metrics.CategoricalCrossentropy()
val_inception_loss = tf.keras.metrics.CategoricalCrossentropy()
acc_fn = accuracy_score

INCEPTION_EPOCHS = 2
INCEPTION_BATCH_SIZE = 32

#  Number of samples for evaluations per batch
INCEPTION_TEST_SAMPLE = 200
