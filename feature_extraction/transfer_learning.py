import openl3
from pathlib import Path
import numpy as np
import librosa
import tensorflow_hub as hub


class AudioL3:

    def __init__(self, input_repr: str = 'mel256', content_type: str = 'music', embedding_size: int = 512) -> None:

        self.model = openl3.models.load_audio_embedding_model(input_repr=input_repr,
                                                              content_type=content_type,
                                                              embedding_size=embedding_size)

    def get_embedding(self,
                      audio: Path,
                      sr: int = 16000,
                      batch_size: int = 1,
                      hop_size: float = 0.5,
                      center: bool = True) -> np.ndarray:
        # Read audio
        y, _ = librosa.load(audio, sr=sr)

        # Calculate embedding
        emb, ts = openl3.get_audio_embedding(y,
                                             sr,
                                             model=self.model,
                                             batch_size=batch_size,
                                             hop_size=hop_size,
                                             center=center,
                                             verbose=False)
        emb = emb[1:]
        emb = np.mean(emb, axis=0)

        return emb


class YamNet:

    def __init__(self) -> None:
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    def get_embedding(self, audio: Path, sr: int = 16000) -> np.ndarray:
        # Read audio
        y, _ = librosa.load(audio, sr=sr)

        # Calculate embedding
        _, emb, _ = self.model(y)
        emb = np.mean(emb, axis=0)

        return emb
