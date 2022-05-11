import tensorflow as tf 

class Generator:
	def __init__(self, shape, init):
		self.shape = shape 
		self.init = init 

	def upsample(self, inputs, skips, filters):
		init = tf.keras.initializers.RandomNormal(stddev=0.02)
		z = tf.keras.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same',
						kernel_initializer=init)(inputs)
		z = tf.keras.layers.BatchNormalization()(z, training=True)
		z = tf.keras.layers.Concatenate()([z, skips])
		z = tf.keras.layers.Activation('relu')(z)
		return z

	def downsample(self, inputs, filters, bn=True):
		init = tf.keras.initializers.RandomNormal(stddev=0.02)
		z = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=init)(inputs)
		if bn:
			z = tf.keras.layers.BatchNormalization()(z, training=True)
		z = tf.keras.layers.LeakyReLU(0.2)(z)
		return z

	def build_model(self):
		image = tf.keras.layers.Input(shape=self.shape)
		e1 = self.downsample(image, 64, bn=False)
		e2 = self.downsample(e1, 128)
		e3 = self.downsample(e2, 256)
		e4 = self.downsample(e3, 512)
		b = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same',
					kernel_initializer=self.init)(e4)
		b = tf.keras.layers.Activation('relu')(b)
		d4 = self.upsample(b, e4, 512)
		d5 = self.upsample(d4, e3, 256)
		d6 = self.upsample(d5, e2, 128)
		d7 = self.upsample(d6, e1, 64)
		z = tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same',
						kernel_initializer=self.init)(d7)
		out = tf.keras.layers.Activation('tanh')(z)
		generator = tf.keras.models.Model(image, out)
		return generator
