import numpy as np
import pathlib
import librosa
import scipy
from feature_extraction.audio import AudioReader


class STFT:

    def __init__(self, **kwargs):
        # Spectrogram options
        self.n_fft = 2048
        self.window_type = 'hamming_asymmetric'
        self.spec_type = 'magnitude'
        self.win_len = 0.04
        self.hop_len = 0.02
        self.center = True
        self.win_type_flag = 'time'

        # Audio-read options
        self.sr = 32000
        self.channel = 'mono'
        self.duration = None

        for key, value in kwargs.items():
            if key in dir(self):
                if isinstance(value, type(vars(self)[key])):
                    vars(self)[key] = value
                else:
                    raise TypeError(f'Input kwarg [{key}] type is not correct. (Needed: {type(vars(self)[key])})')

    def calculate_stft(self, input_arg):
        """
        Computes Short-Time Fourier Transform
        Args:
            input_arg (Union [pathlib.Path | np.ndarray): Path to audio or audio vector
        Returns:
            S (np.ndarray): STFT of given audio
        """

        eps = np.spacing(1)

        # Read audio and store in 'y' variable
        if isinstance(input_arg, pathlib.Path):
            path = input_arg
            audio_reader = AudioReader(output_sr=self.sr, channel=self.channel, duration=self.duration)
            y = audio_reader.read(path)
        elif isinstance(input_arg, np.ndarray):
            y = input_arg
        else:
            raise TypeError(f'Unknown input type [{type(input_arg)}] for calculating STFT')

        # Set window and hop lengths in samples
        self.pass_win_hop_to_samples()

        # Window creation
        window = self.window_function(self.win_len)

        # STFT Calculation
        # spec.shape[0] = n_fft // 2 + 1
        # spec.shape[1] = (new_y_length - win_len)/(win_len - hop_len) + 1
        # np.pad (reflect mode): [n_fft//2, y, n_fft//2] -> new_y_length = len(y) + n_fft
        if self.spec_type == 'magnitude':
            spec = np.abs(librosa.stft(y + eps,
                                       n_fft=self.n_fft,
                                       win_length=self.win_len,
                                       hop_length=self.hop_len,
                                       window=window,
                                       center=self.center))
        elif self.spec_type == 'power':
            spec = np.abs(librosa.stft(y + eps,
                                       n_fft=self.n_fft,
                                       win_length=self.win_len,
                                       hop_length=self.hop_len,
                                       window=window,
                                       center=self.center)) ** 2
        else:
            raise ValueError(f'Unknown spectrum type [{self.spec_type}]')

        return spec

    def pass_win_hop_to_samples(self):
        """
            Set window and hop lengths in samples
            Returns:
                None
            """

        if self.win_type_flag == 'time':
            self.win_len = int(self.win_len * self.sr)
            self.hop_len = int(self.hop_len * self.sr)
            self.win_type_flag = 'sample'
        elif self.win_type_flag == 'sample':
            pass
        else:
            raise ValueError(f'{self.win_type_flag} is not a valid value. Expected time or sample')

    def check_input_type(self, input_arg):
        """
        Check input_arg. If it is a path or audio array, it calculates STFT. If not, it is considered as STFT
        Args:
            input_arg (Union[pathlib.Path, np.nadarray): Path to audio, audio array or STFT
        Returns:
            spec (np.ndarray): Short-Time Fourier Transform
        """

        if isinstance(input_arg, pathlib.Path):
            spec = self.calculate_stft(input_arg)
        elif isinstance(input_arg, np.ndarray):
            if len(input_arg.shape) == 1 or (len(input_arg.shape) == 2 and input_arg.shape[1] <= 2):
                spec = self.calculate_stft(input_arg)
            else:
                spec = input_arg
        else:
            message = f'Unknown input type [{type(input_arg)}] for extracting features'
            raise TypeError(message)

        return spec

    def window_function(self, n):
        if self.window_type == 'hamming_asymmetric':
            return scipy.signal.hamming(n, sym=False)

        elif self.window_type == 'hamming_symmetric' or self.window_type == 'hamming':
            return scipy.signal.hamming(n, sym=True)

        elif self.window_type == 'hann_asymmetric':
            return scipy.signal.hann(n, sym=False)

        elif self.window_type == 'hann_symmetric' or self.window_type == 'hann':
            return scipy.signal.hann(n, sym=True)

        else:
            message = 'Unknown window type [{}]'.format(self.window_type)

            raise ValueError(message)


class MelSpectrogram(STFT):

    def __init__(self, **kwargs):
        STFT.__init__(self, **kwargs)  # Check if kwargs(logmel kwargs + stft_kwargs) don't rise error

        # Spectrogram options
        self.n_bands = 64
        self.fmin = 0
        self.fmax = None
        self.normalize_mel_bands = True
        self.logarithmic = True
        self.htk = False

        for key, value in kwargs.items():
            if isinstance(value, type(vars(self)[key])):
                vars(self)[key] = value
            else:
                raise TypeError(f'Input kwarg [{key}] type is not correct. (Needed: {type(vars(self)[key])})')

    def extract(self, input_arg):
        """
        Computes (Log)-Mel Spectrogram
        Args:
            input_arg (Union [pathlib.Path | np.ndarray): Path to audioaudio vector
        Returns:
            S (np.ndarray): STFT of given audio
        """
        eps = np.spacing(1)

        spec = self.check_input_type(input_arg)

        # Set fmax if None
        if self.fmax is None:
            self.fmax = int(self.sr / 2)

        # Create Mel filterbank
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_bands,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk)

        # Normalize mel bands
        if self.normalize_mel_bands:
            mel_basis /= np.max(mel_basis, axis=-1)[:, None]

        # Apply Mel filterbank to
        mel_spec = np.dot(mel_basis, spec)

        # Log mel spectrogram if it is specified
        if self.logarithmic:
            mel_spec = np.log(mel_spec + eps)

        return mel_spec
