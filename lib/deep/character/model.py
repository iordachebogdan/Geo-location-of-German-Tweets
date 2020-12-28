from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Convolution1D,
    MaxPooling1D,
    ThresholdedReLU,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


class CharacterLevelCNN:
    def __init__(
        self,
        input_size,
        alphabet_size,
        embedding_size,
        conv_layers,
        fully_connected_layers,
        num_of_classes,
        threshold,
        dropout_p,
        optimizer="adam",
        loss=None,
    ):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.optimizer = Adam(learning_rate=0.01)
        self.loss = loss or (
            "categorical_crossentropy" if num_of_classes > 1 else "MSE"
        )
        self._build()

    def _build(self):
        inputs = Input(shape=(self.input_size,), name="sent_input", dtype="int64")
        x = Embedding(
            self.alphabet_size + 1, self.embedding_size, input_length=self.input_size
        )(inputs)
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)
        predictions = Dense(
            self.num_of_classes,
            activation=("softmax" if self.num_of_classes > 1 else None),
        )(x)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model = model
        print("CharacterLevelCNN model built: ")
        self.model.summary()

    def train(
        self,
        training_inputs,
        training_labels,
        validation_inputs,
        validation_labels,
        epochs,
        batch_size,
    ):
        self.model.fit(
            training_inputs,
            training_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=epochs,
            batch_size=batch_size,
        )

    def predict(self, testing_inputs, batch_size):
        return self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)
