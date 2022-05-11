import progressbar
import tensorflow as tf

GEN_SHAPE = (128, 128, 3)
DISC_SHAPE = (128, 128, 6)

disc_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gen_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

disc_loss = tf.keras.losses.BinaryCrossentropy()
gen_loss1 = tf.keras.losses.BinaryCrossentropy()
gen_loss2 = tf.keras.losses.MeanAbsoluteError()

widgets = ["Epoch %s/%s ", progressbar.Percentage(), " ", progressbar.Bar(),
           " ", progressbar.ETA(), " ", progressbar.Variable("g_loss"),
           " ", progressbar.Variable("d_loss")]

BATCH_SIZE = 128
N_EPOCHS = 40

# 0.0 as default value
g_loss, d_loss = 0.0, 0.0

N_COLS, N_ROWS = 7, 4
