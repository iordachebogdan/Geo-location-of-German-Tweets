from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    GRUCell,
    RNN,
    StackedRNNCells,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf


class StackedGRU:
    def __init__(
        self,
        train_texts,
        num_of_classes=1,
        vocab_size=10000,
        embedding_size=600,
        hidden_dim=600,
        num_layers=3,
        dropout_p=0,
        clf_dim=[1024],
        optimizer="adam",
        loss=None,
        word_embeddings=None,
    ):
        """Parameters:
        train_texts: list of strings
        num_of_classes: int (1 if regression)
        vocab_size: int (maximum number of words in vocabulary)
        embedding_size: int (word embeddings size)
        hidden_dim: int (hidden dimension for GRU)
        num_layers: int (number of stacked GRU layers)
        dropout_p: float (dropout probability)
        clf_dim: list of ints (sizes of dense layers after the GRU)
        optimizer: str (optimizer to use)
        loss: str (name of loss function to use)
        word_embeddings: None or Word2VecEmbeddings object
        """
        self.num_of_classes = num_of_classes
        self.optimizer = optimizer
        self.loss = loss or (
            "categorical_crossentropy" if num_of_classes > 1 else "MSE"
        )
        self.encoder = TextVectorization(max_tokens=vocab_size, standardize=None)
        print("Fitting text vectorizer...")
        self.encoder.adapt(train_texts)
        print("Building model...")
        sequence = [
            Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=embedding_size,
                mask_zero=True,
                weights=(
                    None
                    if word_embeddings is None
                    else [word_embeddings.get_emb_matrix(self.encoder)]
                ),
                trainable=(word_embeddings is None),
            ),
        ]
        # stack GRU Cells
        gru_cells = [GRUCell(hidden_dim) for _ in range(num_layers)]
        stacked_gru_cells = StackedRNNCells(gru_cells)
        gru_layer = RNN(stacked_gru_cells)
        sequence.append(gru_layer)
        for dim in clf_dim:
            sequence.append(Dense(dim, activation="relu"))
            sequence.append(Dropout(dropout_p))
        sequence.append(
            Dense(
                self.num_of_classes,
                activation=("softmax" if self.num_of_classes > 1 else None),
            )
        )
        self.model = Sequential(sequence)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=([] if self.num_of_classes == 1 else ["accuracy"]),
        )

    def train(
        self,
        training_inputs,
        training_labels,
        validation_inputs,
        validation_labels,
        epochs,
        batch_size,
        es,
    ):
        training_inputs = self.encoder(training_inputs)
        validation_inputs = self.encoder(validation_inputs)
        training_labels = tf.convert_to_tensor(training_labels, dtype=tf.float32)
        validation_labels = tf.convert_to_tensor(validation_labels, dtype=tf.float32)
        # use early stopping
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=es)
        # save best model
        mc = ModelCheckpoint(
            "checkpoints/best.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
        )
        self.model.fit(
            training_inputs,
            training_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, mc],
        )
        self.model = load_model("checkpoints/best.h5")

    def predict(self, testing_inputs, batch_size):
        testing_inputs = self.encoder(testing_inputs)
        return self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)
