import tensorflow as tf


class InceptionV3:
    def __init__(self, classes, freeze_layers, shape):
        self.classes = classes
        self.freeze_layers = freeze_layers 
        self.shape = shape 
        
    def build_model(self):
        inceptionv3 = tf.keras.applications.InceptionV3(include_top=False,
                                                        weights="imagenet",
                                                        input_tensor=tf.keras.layers.Input(self.shape))
        x = tf.keras.layers.GlobalAveragePooling2D()(inceptionv3.output)
        x = tf.keras.layers.Dense(self.classes, activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inceptionv3.inputs, outputs=x)
        for i in range(self.freeze_layers):
            model.layers[i].trainable = False
        return model, tf.keras.applications.preprocess_input