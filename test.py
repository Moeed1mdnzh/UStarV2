import tensorflow as tf 


class Discriminator:
	def __init__(self, shape, init):
		self.shape = shape
		self.init = init  

	def build_model(self): 
		image = tf.keras.layers.Input(shape=self.shape)
		x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=self.init)(image)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=self.init)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=self.init)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=self.init)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Flatten()(x)
		out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
		discriminator = tf.keras.models.Model(image, out)
		return discriminator