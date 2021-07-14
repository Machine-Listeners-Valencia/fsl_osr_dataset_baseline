from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np

class BaselineModel:

    def __init__(self, embedding_size: int, number_of_classes: int) -> None:
        self.embedding_size = embedding_size
        self.number_of_classes = number_of_classes

        # Model creation
        self.model = self.create_model()

        # LR on plateau
        self.lr_plateau = self.lr_onplateau()

        # Early stopping
        self.early_stopping = self.early_stopping_function()

    def create_model(self) -> Model:
        # Declaration of layer and interaction
        inp = Input(shape=(self.embedding_size,))
        x = Dense(512, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.number_of_classes, activation='sigmoid')(x)

        # Declaration of model
        model = Model(inputs=inp, outputs=x)

        # Optimizer declaration
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

        return model

    @staticmethod
    def lr_onplateau(monitor: str = 'val_categorical_accuracy',
                     factor: float = 0.75,
                     patience: int = 20,
                     min_lr: float = 0.000001
                     ) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr)

    @staticmethod
    def early_stopping_function(monitor: str = 'val_categorical_accuracy',
                                min_delta: float = 0.00001,
                                mode: str = 'auto',
                                patience: int = 50
                                ) -> EarlyStopping:
        return EarlyStopping(monitor=monitor, mode=mode, min_delta=min_delta, patience=patience)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 200,
              batch_size: int = 16
              ) -> None:
        self.model.fit(X_train,
                       y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[self.lr_plateau, self.early_stopping],
                       verbose=0)

    def openset_predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X_test)
        openset_predictions = self.predict_on_threshold(predictions)

        return openset_predictions

    @staticmethod
    def predict_on_threshold(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        predict_openset = np.zeros((predictions.shape[0], predictions.shape[1]))

        for j in range(predictions.shape[0]):
            max_value = np.amax(predictions[j, :])
            idx_max_value = np.where(predictions[j, :] == max_value)
            idx_max_value = idx_max_value[0]

            if max_value >= threshold:
                predict_openset[j, idx_max_value] = 1

        return predict_openset
