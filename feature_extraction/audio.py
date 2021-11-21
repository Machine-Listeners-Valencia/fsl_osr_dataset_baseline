from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


class AudioReader:
    """
    Class that reads audios

    Attributes:
        output_sr (int): Output audios sampling rate
        chanel (str): Desirable channel to read
        duration (Union[int, NoneType]): Length of the output audio in seconds

    """

    def __init__(self, output_sr=48000, channel='mono', duration=None, norm=True):
        """
        init method

        Args:
            output_sr (int): Output audios sampling rate. Default: 48000
            channel (str): Desirable channel to read. Possible_values: ['mono', 'right', 'left', 'diff' (left-right)]
                           Default: 'mono'
            duration (Union[int, NoneType]): Length of the audio. Default: None
        """

        # Check allowed variable types
        if not isinstance(output_sr, int):
            raise TypeError(f'output_sr must be of type int')
        if not isinstance(channel, str):
            raise TypeError(f'channel must be of type str')
        if not isinstance(duration, (int, type(None))):
            raise TypeError(f'duration must be of type int or None')

        # Check allowed channel options
        available_channels = ['mono', 'left', 'right', 'diff', 'stereo']
        if channel not in available_channels:
            raise ValueError(f'channel must be one of the following options: {available_channels}')

        self.output_sr = output_sr
        self.channel = channel
        self.duration = duration
        self.norm = norm

    def read(self, path):
        """
        Read an audio given a path

        Args:
            path (pathlib.Path): path of the audio to read

        Returns:
            y (np.ndarray): array of audio values
        """
        if not isinstance(path, Path):
            raise TypeError(f'path must be of type pathlib.Path')

        if path.suffix == '.flac':
            y, original_sr = librosa.load(path, sr=self.sr)
        else:
            y, original_sr = sf.read(file=path, start=None, stop=None)

        y = np.asfortranarray(y)

        y = self.resample_to_sr(y, original_sr)

        # clip duration if the self.duration is not None
        if self.duration is not None:
            if len(y.shape) > 1:
                if (self.duration * self.output_sr) < y.shape[0]:
                    y = y[:(self.duration * self.output_sr), :]
                else:
                    pad = np.zeros(((self.duration * self.output_sr) - y.shape[0],y.shape[1]))
                    y = np.concatenate((y, pad))
            else:
                if (self.duration * self.output_sr) < y.shape[0]:
                    y = y[:(self.duration * self.output_sr)]
                else:
                    pad = np.zeros(((self.duration * self.output_sr) - y.shape[0],))
                    y = np.concatenate((y, pad))

        y = self.select_channel(y)

        if self.norm:
            y = librosa.util.normalize(y)

        return y

    def resample_to_sr(self, y, original_sr):
        """
        If original sampling rate does not match output sampling rate

        Args:
            y (np.ndarray): Audio array with shape according to original sampling rate
            original_sr (int): Original sampling rate

        Returns:
            y (np.ndarray): Audio array with shape according to output sampling rate
        """

        if self.output_sr != original_sr:
            if len(y.shape) == 2:
                y0 = librosa.core.resample(y=y[:, 0], orig_sr=original_sr, target_sr=self.output_sr)
                y1 = librosa.core.resample(y=y[:, 1], orig_sr=original_sr, target_sr=self.output_sr)

                y = np.concatenate((y0, y1), axis=1)
            else:
                y = librosa.core.resample(y=y, orig_sr=original_sr, target_sr=self.output_sr)

        return y

    def select_channel(self, y):
        """
        Extract specified channels from audio array

        Args:
            y (np.ndarray): Audio array with original channels

        Returns:
            y (np.ndarray): Audio array with output channels
        """
        if self.channel == 'mono':
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
        else:
            if len(y.shape) == 1:
                raise ValueError(f'For mono audios, only mono read is allowed')
            elif self.channel == 'left':
                y = y[:, 0]
            elif self.channel == 'right':
                y = y[:, 1]
            elif self.channel == 'diff':
                y = y[:, 0] - y[:, 1]
            elif self.channel == 'stereo':
                y = y

        return y
