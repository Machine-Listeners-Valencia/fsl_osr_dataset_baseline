import os

from typing import Tuple

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import tensorflow.keras as keras

from sklearn.linear_model import LogisticRegression

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


class OpenSetDCAE:

    def __init__(self, number_of_classes: int) -> None:
        self.nceps = 64
        self.feat_length = 200
        self.number_of_classes = number_of_classes

        # CNN Model
        self.model_cnn = self.create_cnn_model()

        # DCAE Model
        #self.dcae_model = self.create_dcae_model()

        # LR on plateau
        self.lr_plateau = self.lr_onplateau()

        # Early stopping
        self.early_stopping = self.early_stopping_function()

    def create_cnn_model(self) -> Model:
        """
        Definition of the 2D CNN used for mel spectrograms.
        :param feat_length: length of the features being used
        :param nceps: Number of mel filters.
        :param trainable: Should the layers be trainable or constant
        :param name: Name of the model , important when reloading the weights
        :return: CNN model
        """
        input_model = keras.layers.Input(shape=(self.nceps, self.feat_length, 1), dtype='float32')

        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(input_model)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

        x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

        x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        model_output = keras.layers.Dense(self.number_of_classes, activation='softmax')(x)

        cnn_model = keras.Model(inputs=[input_model], outputs=[model_output])
        cnn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(decay=0.0001),
                          metrics=['accuracy'])

        return cnn_model

    def create_dcae_model(self) -> Model:
        """
        Definition of the 2D DCAE used for mel spectrograms.
        :param feat_length: length of the features being used
        :param nceps: Number of mel filters.
        :param trainable: Should the layers be trainable or constant
        :param name: Name of the model , important when reloading the weights
        :return: CNN model
        """
        input_model = keras.layers.Input(shape=(self.nceps, self.feat_length, 1), dtype='float32')

        # encoder
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(input_model)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

        # decoder
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        #x = keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(x)

        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

        model_output = keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='relu')(x)

        dcae_model = keras.Model(inputs=[input_model], outputs=[model_output])
        dcae_model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(decay=0.0001),
                           metrics=['accuracy'])

        return dcae_model

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
        # self.model.fit(X_train,
        #                y_train,
        #                validation_data=(X_val, y_val),
        #                epochs=epochs,
        #                batch_size=batch_size,
        #                callbacks=[self.lr_plateau, self.early_stopping],
        #                verbose=0)
        X_train =  np.expand_dims(X_train, axis=-1)
        X_val =  np.expand_dims(X_val, axis=-1)
        outlier_scores_eval = [None] * self.number_of_classes
        outlier_labels_eval = [None] * self.number_of_classes
        for i in range(self.number_of_classes):
            
            # FALTA LIDIAR COMO TRATAR OPENNESS 0 Y OPENNESS 100
            X_class_train, y_class_train, X_class_val, y_class_val = self.select_class(X_train, y_train, X_val, y_val, i)
            X_unwanted_train, y_unwanted_train, X_unwanted_val, y_unwanted_val = self.select_unwatend(X_train, y_train, X_val, y_val)
            self.dcae_model = self.create_dcae_model()
            self.dcae_model.fit(X_class_train,
                                X_class_train,
                                validation_data=(X_class_val, X_class_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[self.lr_plateau, self.early_stopping],
                                verbose=0)
            self.dcae_model.save(os.path.join(os.getcwd(),f'dcae_{i}.h5'))
            
            outlier_scores_eval_kk = np.zeros((X_class_val.shape[0]))
            outlier_scores_eval_uu = np.zeros((X_unwanted_val.shape[0]))
            
            for ii in range(0, X_class_val.shape[0]):
                outlier_scores_eval_kk[ii] = self.dcae_model.evaluate(np.expand_dims(X_class_val[ii], axis=0), 
                                                                      np.expand_dims(X_class_val[ii], axis=0))[0]
                
            for ii in range(0, X_unwanted_val.shape[0]):
                outlier_scores_eval_uu[ii] = self.dcae_model.evaluate(np.expand_dims(X_unwanted_val[ii], axis=0), 
                                                                      np.expand_dims(X_unwanted_val[ii], axis=0))[0]
            
            #outlier_scores_eval[i] = self.dcae_model.evaluate(X_class_val, X_class_val)
            outlier_scores_eval[i] = np.concatenate((outlier_scores_eval_kk, outlier_scores_eval_uu))
            outlier_labels_eval[i] = np.concatenate((np.ones(X_class_val.shape[0]), np.zeros(X_unwanted_val.shape[0])))
            
        clf = LogisticRegression(class_weight='balanced')
        feat = np.concatenate(outlier_scores_eval, axis=0)
        feat = np.reshape(feat, (feat.shape[0], 1))
        labels = np.concatenate(outlier_labels_eval, axis=0)

        clf.fit(feat, labels)
        
        # TRAIN CLOSE-SET
        
        X_known_train, y_known_train, X_known_val, y_known_val = self.select_known(X_train, y_train, X_val, y_val)
        
        self.model_cnn.fit(X_known_train,
                        y_known_train,
                        validation_data=(X_known_val, y_known_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[self.lr_plateau, self.early_stopping],
                        verbose=0)
        
        # HACER LA LOGICA?
            

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

    def select_class(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     index_class: int,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_train_index = np.where(y_train==1)[1]
        y_val_index = np.where(y_val==1)[1]
        idx_train = np.where(y_train_index == index_class)[0]
        idx_val = np.where(y_val_index == index_class)[0]

        return X_train[idx_train, :], y_train[idx_train, :], X_val[idx_val, :], y_val[idx_val, :]
    
    def select_unwatend(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        idx_train = np.where(~y_train.any(axis=1))[0]
        idx_val = np.where(~y_val.any(axis=1))[0]
        
        return X_train[idx_train, :], y_train[idx_train, :], X_val[idx_val, :], y_val[idx_val, :]
    
    def select_known(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        idx_train = np.where(y_train.any(axis=1))[0]
        idx_val = np.where(y_val.any(axis=1))[0]
        
        return X_train[idx_train, :], y_train[idx_train, :], X_val[idx_val, :], y_val[idx_val, :]