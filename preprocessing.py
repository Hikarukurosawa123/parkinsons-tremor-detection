from data_loader import DataLoader
import numpy as np
from scipy.signal import lfilter, firwin, filtfilt, savgol_filter, find_peaks
import scipy
from scipy.signal.windows import hamming
from spectrum import arma2psd, arburg
from pylab import log10, pi, plot, xlabel, randn
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

class AccelerometerData(DataLoader):
    def __init__(self, file_path, frequency):
        super().__init__(
            file_path,
            meta_data={
                "freq(Hz)": frequency,
            },
        )
        self.features = [] # (start, end, duration)

    def print_data(self):
        # Access the data
        print("Accelerometer data shape:", self.data.shape)
        print("Accelerometer data:", self.data)

    # Remove drift
    def _remove_drift(self, window_size=50):
        """
        Removes drift using a moving average filter.
        Args:
            data: Array of accelerometer data.
            window_size: Size of the moving average window.
        Returns:
            Drift-removed data.
        """
        b = np.ones(window_size) / window_size
        a = [1]
        self.data = lfilter(b, a, self.data)

    # Bandpass filter
    def _bandpass_filter(self, lowcut=1.0, highcut=30.0, fs=100):
        """
        Filters the signal to only retain components in the 1-30 Hz range.
        Args:
            data: Array of accelerometer data.
            fs: Sample frequency.
            lowcut: Lower cut-off frequency.
            highcut: Upper cut-off.
        Returns:
            Bandpass filtered data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b = firwin(numtaps=101, cutoff=[low, high], pass_zero=False)
        self.data = filtfilt(b, [1.0], self.data)

    # Windowing with Hamming Function
    def segment_data(self, window_size, overlap_ratio):
        """
        Segments data into overlapping windows with Hamming weights.
        Args:
            data: Array of accelerometer data.
            window_size: Number of samples per window.
            overlap_ratio: Fraction of overlap between consecutive windows.
        Returns:
            Array of segmented windows.
        """
        step_size = int(window_size * (1 - overlap_ratio))
        n_samples = self.data.shape[1]
        n_windows = (n_samples - window_size) // step_size + 1
        hamming_window = hamming(window_size)
        segments = np.zeros((self.data.shape[0], window_size, n_windows))
        
        for x in range(self.data.shape[0]):
            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                segments[x, :, i] = self.data[x, start_idx:end_idx] * hamming_window
                
        self.data = segments

    def detect_peak_frequency(self, fs=100, low_freq=3, high_freq=8, ar_order=6):
        """
        Detects peak frequency using an autoregressive model.
        Args:
            window: Windowed segment of accelerometer data.
            fs: Sampling frequency.
            low_freq: Lower frequency bound for tremor detection.
            high_freq: Upper frequency bound for tremor detection.
            ar_order: Order of the autoregressive model.
        Return:
            Peak frequency if within tremor range, otherwise 1.
        """
        new_data = [[], [], []]
        for i in range(3):
            for j in range(self.data.shape[2]):
                d = self.data[i, :, j]
                [A, P, K] = arburg(d, ar_order)
                
                # Generate the frequency response from the AR coefficients
                freqs = np.linspace(
                    0, fs / 2, 512
                )  # Frequency range up to Nyquist frequency
                #_, h = np.linalg.eigh(
                #    np.polyval(A, np.exp(-1j * 2 * np.pi * freqs / fs))
                #)

                PSD = arma2psd(A, rho=P, NFFT=512)
                PSD = PSD[len(PSD):len(PSD)//2:-1]
                xf = np.linspace(0, 1, len(PSD)) * 100 / 2 #multiply normalized frequency by Nyquist frequency 
                PSD = (np.abs(PSD) ** 2)

                # Normalize the power spectral density (PSD)
                #PSD /= np.sum(PSD)

                # Find the frequency bin corresponding to the maximum power
                peaks, _ = find_peaks(PSD)
                if peaks.size == 0:
                    new_data[i].append(1)  # No peaks detected
                    continue
    
                # Convert peak indices to frequencies
                peak_freqs = xf[peaks]

                # Filter peaks within the desired frequency range
                tremor_peaks = peak_freqs[(peak_freqs >= low_freq) & (peak_freqs <= high_freq)]
                if tremor_peaks.size > 0:
                    PSD_indices = xf == tremor_peaks

                #obtain peak power to filter out low vibrations 
                peak_max = np.max(PSD[peaks])
                
                # Return the dominant tremor frequency, if any, with sufficient power magnitude
                if tremor_peaks.size > 0 and (np.max(PSD[PSD_indices]) > peak_max / 10):
                    xf_PSD_indices = xf[PSD_indices]
                    new_data[i].append(xf_PSD_indices[np.argmax(PSD[PSD_indices])])
                else:
                    new_data[i].append(1)
                    # No peak within the tremor ra
        
        self.data = new_data

    def _smooth_data(self, window_size=50):
        self.data = savgol_filter(self.data, window_size, 3)
        # self.data = np.convolve(self.data, np.ones(window_size)/window_size, mode='same')

    def _multiply(self):
        self.data = self.data[0] * self.data[1] * self.data[2]
    

    def _thresholding(self, threshold=3.5):
        threshold_mask = self.data > threshold 
        self.data[threshold_mask] = 1
        self.data[~threshold_mask] = 0

    def check_duration(self, duration = 300):
        window_beg = 0
        window_end = window_beg + duration

        time_stamp = np.zeros(len(self.data))
        while window_end < len(self.data):
            if not np.sum(self.data[window_beg:window_end] == 0):
                time_stamp[window_beg:window_end] = 1
                window_beg = window_end
                
            window_beg += 1
            window_end = window_beg + duration

        return time_stamp 



    def _feature_extraction(self, threshold=3):
        start = None
        for i in range(self.data.shape[0]):
            if self.data[i] == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    if end - start + 1 > threshold:
                        self.features.append((start, end, end - start + 1))
                    start = None

    def resample(self, size): 
        self.data = scipy.signal.resample(self.data, size, axis = -1)

    def preprocess_data(self):
        time_series_length = self.data.shape[1]
        self._remove_drift()
        self._bandpass_filter()
        self.segment_data(300, 0.9)
        self.detect_peak_frequency()
        self.resample(time_series_length) 
       
        #### 
        self._smooth_data()
        self._multiply()
        self._thresholding()
        self.data = self.check_duration()

    def plot_data(self, t_start=0, t_end=None):
        import matplotlib.pyplot as plt

        time = np.arange(self.data.shape[1]) / self.meta_data["freq(Hz)"]

        # Set t_end to the end of the data if not provided
        if t_end is None:
            t_end = time[-1]

        # Select the data in the specified time range
        start_idx = int(t_start * self.meta_data["freq(Hz)"])
        end_idx = int(t_end * self.meta_data["freq(Hz)"])
        time_range = time[start_idx:end_idx]
        data_range = self.data[:, start_idx:end_idx]

        plt.subplot(3, 1, 1)
        plt.plot(time_range, data_range[0], "r")
        plt.title("Accelerometer Data - X Axis")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time_range, data_range[1], "g")
        plt.title("Accelerometer Data - Y Axis")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time_range, data_range[2], "b")
        plt.title("Accelerometer Data - Z Axis")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
