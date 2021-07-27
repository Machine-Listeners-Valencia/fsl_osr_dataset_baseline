import sys
import pickle
import tensorflow

if sys.argv[1] == 'l3':
    from feature_extraction.transfer_learning import AudioL3
    extractor = AudioL3()
elif sys.argv[2] == 'yamnet':
    from feature_extraction.transfer_learning import YamNet
    extractor = YamNet()
else:
    raise Exception('Not available pre-trained network')
    print('Not available pre-trained network')

if sys.argv[1] == 'l3':
    emb = extractor.get_embedding(sys.argv[2])

model = pickle.load(open(sys.argv[3],"rb"))

preds = model.openset_predict(emb)

print(preds)
