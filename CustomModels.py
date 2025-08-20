import tensorflow as tf
from tensorflow.keras import layers, models
class DeepMLP_3(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_3, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(512, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_3, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_3 built with input shape:", input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeepMLP_5(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_5, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(1024, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_5, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_5 built with input shape:", input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeepMLP_7(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_7, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(4096, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(2048, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1024, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_7, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_7 built with input shape:", input_shape)
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
class AutoencoderClassifier(models.Model):
    def __init__(self, input_size, encoding_dim, num_classes, _activation='relu', **kwargs):
        super(AutoencoderClassifier, self).__init__(**kwargs)
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self._activation = _activation
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(encoding_dim, activation=_activation)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation=_activation, input_shape=(encoding_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(input_size, activation='sigmoid')
        ])
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation=_activation, input_shape=(encoding_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(AutoencoderClassifier, self).build(input_shape)
        encoded_shape = self.encoder.compute_output_shape(input_shape)
        print("AutoencoderClassifier built with input shape:", input_shape,
              "and encoder output shape:", encoded_shape)
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return {"decoder": decoded, "classifier": classification}

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "encoding_dim": self.encoding_dim,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)