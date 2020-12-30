from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


class BiLSTM:
    def __init__(
        self,
        train_texts,
        num_of_classes=1,
        vocab_size=10000,
        embedding_size=600,
        hidden_dim=600,
        clf_dim=1024,
        dropout_p=0.5,
        optimizer="adam",
        loss=None,
    ):
        self.num_of_classes = num_of_classes
        self.optimizer = optimizer
        self.loss = loss or (
            "categorical_crossentropy" if num_of_classes > 1 else "MSE"
        )
        encoder = TextVectorization(max_tokens=vocab_size, standardize=None)
        print("Fitting text vectorizer...")
        encoder.adapt(train_texts)
        print("Building model...")
        sequence = [
            encoder,
            Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=embedding_size,
                mask_zero=True,
            ),
            Bidirectional(LSTM(hidden_dim)),
        ]
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
    ):
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
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
        return self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)
