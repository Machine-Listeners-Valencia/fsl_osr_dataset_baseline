import sys

import tensorflow

if sys.argv[1] == 'l3net':
    from feature_extraction.transfer_learning import AudioL3
    extractor = AudioL3()
elif sys.argv[2] == 'yamnet':
    from feature_extraction.transfer_learning import YamNet
    extractor = YamNet()
else:
    raise Exception('Not available pre-trained network')
    print('Not available pre-trained network')

if sys.argv[1] == 'l3net':
    emb = extractor.get_embedding(sys.argv[2])

model = tensorflow.keras.models.load_model(sys.argv[3])

model.predict(emb)

print(emb)
