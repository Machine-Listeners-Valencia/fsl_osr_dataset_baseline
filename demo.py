import sys
import numpy as np
import tensorflow
from tensorflow.keras.models import Model


def openset_predict(model: Model, X_test: np.ndarray) -> np.ndarray:
    predictions = model.predict(X_test)
    openset_predictions = predict_on_threshold(predictions)

    return openset_predictions


def predict_on_threshold(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predict_openset = np.zeros((predictions.shape[0], predictions.shape[1]))

    for j in range(predictions.shape[0]):
        max_value = np.amax(predictions[j, :])
        idx_max_value = np.where(predictions[j, :] == max_value)
        idx_max_value = idx_max_value[0]

        if max_value >= threshold:
            predict_openset[j, idx_max_value] = 1

    return predict_openset


if sys.argv[1] == 'l3':
    from feature_extraction.transfer_learning import AudioL3

    extractor = AudioL3()
elif sys.argv[1] == 'yamnet':
    from feature_extraction.transfer_learning import YamNet

    extractor = YamNet()
else:
    raise Exception('Not available pre-trained network')
    print('Not available pre-trained network')

emb = extractor.get_embedding(sys.argv[2])
emb = np.expand_dims(emb, axis=0)
print(emb.shape)

# load model with keras
model = tensorflow.keras.models.load_model(sys.argv[3])

preds = openset_predict(model, emb)

print(preds)
