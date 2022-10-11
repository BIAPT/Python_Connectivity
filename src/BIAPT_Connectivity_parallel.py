import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr, wilcoxon
import multiprocessing as mp

class ConnectivityMeasure:
    '''
    Parent class for both PLI and AEC connectivity measures.

    Parameters
    ----------
    verbose: bool, optional
        Verbose mode for logging information during runtime. Default: False
    '''

    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute(self, data):
        '''
        Used to compute connectivity within each window in child classes.

        Parameters
        ----------
        data: ndarray of shape (windows, channels, time)
            Windowed data.

        Returns
        -------
        conn: ndarray of shape (windows, channels, channels)
            Connectivity data.
        '''
        raise NotImplementedError()



class BasePLI(ConnectivityMeasure):
    '''
    Parent class for both wPLI and dPLI connectivity measures.

    Parameters
    ----------
    n_surrogates: int, optional
        Number of surrogates used in the case of surrogate analysis.
        Default: 20
    verbose: bool, optional
        Verbose mode for logging information during runtime. Default: False
    '''

    def __init__(self, n_surrogates=20, verbose=False):
        super().__init__(verbose=verbose)
        self.n_surrogates = n_surrogates

    def compute(self, data):
        '''
        Compute PLI within each window.

        Parameters
        ----------
        data: ndarray of shape (windows, channels, time)
            Windowed data.

        Returns
        -------
        pli: ndarray of shape (windows, channels, channels)
            PLI connectivity matrix.
        '''

        pli_windows = []

        #Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())

        input = []
        for n, window in enumerate(data):
            input.append((n,window,data))

        # Parallelize this
        pli_windows = pool.starmap(self.calculate_wPLI, input)

        pli = np.array(pli_windows)
        return pli


    def calculate_wPLI(self,n, window, data):
        if self.verbose:
            print(f"Computing on window {n+1}/{len(data)}")
        pli_window = self._pli_window(window, self._pli)
        if self.n_surrogates > 0:
            pli_window = self._correct_pli_window(window, pli_window)
        return pli_window
        # Not Parallelized
        #for n, window in enumerate(data):
        #    if self.verbose:
        #        print(f"Computing on window {n+1}/{len(data)}")
        #    pli_window = self._pli_window(window, self._pli)
        #    if self.n_surrogates > 0:
        #        pli_window = self._correct_pli_window(window, pli_window)
        #    pli_windows.append(pli_window)
        #pli = np.array(pli_windows)
        #return pli

    def _pli_window(self, window, pli_func):
        ch = window.shape[0]
        pli_mat = np.zeros((ch, ch))
        trans_signal = hilbert(window)  # (ch, time)
        for ch1 in range(ch):
            for ch2 in range(ch1):
                y1 = trans_signal[ch1]
                y2 = trans_signal[ch2]
                pli = pli_func(y1, y2)
                pli_mat[ch1, ch2] = pli
        return pli_mat

    def _correct_pli_window(self, window, pli_window):
        ch = window.shape[0]
        pli_surrogates = []

        # Not Parallelized Version
        for _ in range(self.n_surrogates):
            pli_surrogates.append(self._pli_window(
                window, self._pli_surrogate))

        pli_surrogates = np.array(pli_surrogates)
        pli_corrected = np.zeros((ch, ch))
        for ch1 in range(ch):
            for ch2 in range(ch1):
                null_dist = pli_surrogates[:, ch1, ch2]
                pli_value = pli_window[ch1, ch2]
                pli_corrected[ch1, ch2] = self._surrogate_correct(
                    pli_value, null_dist)
        return pli_corrected

    def _pli_surrogate(self, y1, y2):
        splice = np.random.randint(len(y2))
        y2_splice = np.concatenate((y2[splice:], y2[:splice]))
        wpli_surr = self._pli(y1, y2_splice)
        return wpli_surr

    def _pli(self, y1, y2):
        raise NotImplementedError()

    def _surrogate_correct(self, pli_value, null_dist):
        raise NotImplementedError()


class WPLI(BasePLI):
    '''
    wPLI Implementation


    Parameters
    ----------
    n_surrogates: int, optional
        Number of surrogates used in the case of surrogate analysis.
        Default: 20
    verbose: bool, optional
        Verbose mode for logging information during runtime. Default: False
    p_value: float, optional
        P-value used in surrogate testing. Default: 0.05
    '''

    def __init__(self, n_surrogates=20, verbose=False, p_value=0.05):
        super().__init__(n_surrogates=n_surrogates, verbose=verbose)
        self.p_value = p_value

    def compute(self, data):
        '''
        Compute wPLI within each window.

        Parameters
        ----------
        data: ndarray of shape (windows, channels, time)
            Windowed data.

        Returns
        -------
        wpli: ndarray of shape (windows, channels, channels)
            wPLI connectivity matrix.
        '''

        wpli = super().compute(data)
        # Fill up the upper half triangle
        wpli = wpli + wpli.transpose([0, 2, 1])

        return wpli

    def _pli(self, y1, y2):

        csd = y1 * np.conjugate(y2)
        num = np.abs(np.mean(np.imag(csd)))
        den = np.mean(np.abs(np.imag(csd)))

        if den == 0:
            wpli = 0
        else:
            wpli = num / den

        return wpli

    def _surrogate_correct(self, pli_value, null_dist):
        _, p = wilcoxon(null_dist - pli_value)
        if p < self.p_value:
            null_median = np.median(null_dist)
            if pli_value < null_median:
                return 0
            else:
                return pli_value - null_median
        else:
            return 0


class DPLI(BasePLI):
    '''
    dPLI Implementation


    Parameters
    ----------
    n_surrogates: int, optional
        Number of surrogates used in the case of surrogate analysis.
        Default: 20
    verbose: bool, optional
        Verbose mode for logging information during runtime. Default: False
    p_value: float, optional
        P-value used in surrogate testing. Default: 0.05
    '''

    def __init__(self, n_surrogates=20, verbose=False, p_value=0.05):
        super().__init__(n_surrogates=n_surrogates, verbose=verbose)
        self.p_value = p_value

    def compute(self, data):
        '''
        Compute dPLI within each window.

        Parameters
        ----------
        data: ndarray of shape (windows, channels, time)
            Windowed data.

        Returns
        -------
        dpli: ndarray of shape (windows, channels, channels)
            dPLI connectivity matrix.
        '''
        dpli_init = super().compute(data)
        n_channels = dpli_init.shape[1]

        # Fill up the upper half triangle
        dpli = dpli_init - np.transpose(dpli_init, [0, 2, 1])
        i = np.triu_indices(n_channels, k=1)
        j = np.diag_indices(n_channels)
        z = np.zeros((1, n_channels, n_channels))
        z[0, i[0], i[1]] = 1
        dpli += z
        dpli[:, j[0], j[1]] = 0.5
        return dpli

    def _pli(self, y1, y2):
        csd = y1 * np.conjugate(y2)
        return np.mean(np.heaviside(np.imag(csd), 0.5))

    def _surrogate_correct(self, pli_value, null_dist):
        _, p = wilcoxon(null_dist - pli_value)
        if p < self.p_value:
            null_median = np.median(null_dist)
            if null_median == 0.5 and pli_value == 0.5:
                # no effect
                return 0.5
            elif null_median >= 0.5 and pli_value <= 0.5:
                # Correct toward 0
                gap = 0.5 - null_median
                return max(0, pli_value - gap)
            elif null_median <= 0.5 and pli_value >= 0.5:
                # Correct toward 1
                gap = null_median - 0.5
                return min(1, pli_value + gap)
            elif null_median >= 0.5 and pli_value >= 0.5:
                # Correct toward 0.5
                gap = pli_value - null_median
                if gap < 0:
                    return 0.5
                return gap + 0.5
            else:
                # Correct toward 0.5
                gap = pli_value - null_median
                if gap > 0:
                    return 0.5
                return gap + 0.5
        else:
            return 0.5


def window_data(data, window_size, step_size, fs):
    '''
    Split continuous data into windows.

    Parameters
    ----------
    data: ndarray of shape (channels, time)
        Continous data to be windowed.
    window_size: float
        Length of each window in seconds.
    step_size : float
        The increment in seconds by which windows are shifted.
    fs: float
        Sampling rate (Hz).

    Returns
    -------
    windows: ndarray of shape (windows, channels, time)
        Windowed data.

    '''
    windows = []
    for ix in range(0, data.shape[1] - (window_size * fs) + 1,
                    step_size * fs):
        windows.append(data[:, ix:ix + (window_size * fs)])
    windows = np.array(windows)
    return windows


def filter_data(data, fmin, fmax, fs):
    '''
    Bandpass filter continous data.

    Parameters
    ----------
    data: ndarray of shape (channels, time)
        Continous data to be filtered.
    fmin: float
        Lower frequency of interest (Hz).
    fmax: float
        Upper frequency of interest (Hz).
    fs: float
        Sampling rate (Hz).

    Returns
    -------
    data_filt: ndarray of shape (channels, time)
        Filtered data.

    '''
    b, a = butter(1, [fmin, fmax], btype='bandpass', analog=False,
                  output='ba', fs=fs)
    data_filt = filtfilt(b, a, data, padlen=3 * (max(len(a), len(b) - 1)))
    return data_filt


def connectivity_compute(data, window_size=None, step_size=None, fmin=None,
                         fmax=None, fs=None, n_surrogates=0, verbose=True,
                         mode="aec"):
    '''
    Generic Function used to compute AEC, dPLI, or wPLI.

    Parameters
    ----------
        data: ndarray of shape (channels, time) or (windows, channels, time)
            The data segment to calculate connectivity with.
        window_size: float, optional
            Length of each window in seconds. Unused if data already processed.
            Default: None
        step_size: float, optional
            The increment by which windows are shifted. Unused if data already
            processed. Default: None
        fmin: float, optional
            Lower frequency of interest (Hz). Unused if data already processed.
            Default: None
        fmax: float, optional
            Upper frequency of interest (Hz). Unused if data already processed.
            Default: None
        fs: float, optional
            Sampling rate (Hz). Unused if data already processed.
            Default: None
        n_surrogates: int, optional
            Number of surrogates used in surrogate analysis. Only applies to
            dPLI or wPLI methods. Default: 0
        verbose: bool, optional
            Verbose mode for logging information during runtime. Default: False
        mode: str, optional
            'aec', 'wpli', or 'dpli'. Default: 'aec'

    Returns
    -------
        metric: ndarray of shape (windows, channels, channels)
            Metric (AEC, wPLI, or dPLI)
    '''

    if data.ndim == 2:  # data is continuous and must be filtered/windowed
        data_filt = filter_data(data, fmin, fmax, fs)
        windows = window_data(data_filt, window_size, step_size, fs)
    elif data.ndim == 3:  # data is already processed
        windows = data
    else:
        raise ValueError(
            "Data must be 2D (channels, time) or 3D (windows, channels, time)")

    metrics = {
        "wpli": WPLI,
        "dpli": DPLI,
    }

    metric_class = metrics[mode]
    metric_computer = metric_class(
        n_surrogates=n_surrogates, verbose=verbose)

    metric = metric_computer.compute(windows)

    return metric
