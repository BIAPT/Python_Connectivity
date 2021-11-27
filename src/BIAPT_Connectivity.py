import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr, wilcoxon


class ConnectivityMeasure:

    def __init__(self, window_size, step_size, fmin, fmax, verbose=False):
        self.window_size = window_size
        self.step_size = step_size
        self.fmin = fmin
        self.fmax = fmax
        self.verbose = verbose

    def _window_data(self, data, fs):
        windows = []
        for ix in range(0, data.shape[1] - (self.window_size * fs) + 1, self.step_size * fs):
            windows.append(data[:, ix:ix + (self.window_size * fs)])
        windows = np.array(windows)
        return windows

    def _filter_data(self, data, fs):
        b, a = butter(1, [self.fmin / (fs / 2), self.fmax / (fs / 2)],
                      btype='bandpass', analog=False, output='ba', fs=None)
        data_filt = filtfilt(b, a, data, padlen=3 * (max(len(a), len(b) - 1)))
        return data_filt

    def compute(self, data, fs):
        data_filt = self._filter_data(data, fs)
        return self._window_data(data_filt, fs)


class AEC(ConnectivityMeasure):

    def __init__(self, window_size, step_size, fmin, fmax, verbose=False):
        super().__init__(window_size, step_size, fmin, fmax, verbose=verbose)

    def compute(self, data, fs):
        windows = super().compute(data, fs)
        return self._aec_windows(windows)

    def _aec_windows(self, windows):
        # initialize AEC
        aec = np.zeros((windows.shape[1], windows.shape[1], windows.shape[0]))
        # iterate over the windows
        for index, window in enumerate(windows):
            if self.verbose:
                print(f"Computing on window {index+1}/{len(windows)}")
            aec[:, :, index] = self._aec_pairwise_corrected(
                window, windows.shape[1], 0)
            index += 1
        return np.transpose(((aec + np.transpose(aec, (1, 0, 2))) / 2), [2, 0, 1])

    def _aec_pairwise_corrected(self, data, num_channels, cut_amount):
        '''
        AEC PAIRWISE CORRECTED helper function to calculate the pairwise corrected aec

        input:
            data (numpy.ndarray (channels, time)): the data segment to calculate pairwise corrected aec on
            num_channels (int): number of channels
            cut_amount (float): the amount we need to remove from the hilbert transform

        output:
            aec: a num_region*num_region matrix which has the amplitude envelope
            correlation between two regions
        '''
        # transpose the data
        data = data.T

        # initialize the aec
        aec = np.zeros((num_channels, num_channels))

        # Pairwise leakage correction in window for AEC
        # Loops around all possible ROI pairs
        for region_i in range(num_channels):

            # select the first channel of interest
            y = data[:, region_i]
            y = y.reshape((len(y), 1))

            for region_j in range(num_channels):

                # ignore correlation with itself
                if region_i == region_j:
                    continue

                # select the second channel of interest
                x = data[:, region_j]
                x = x.reshape((len(x), 1))

                # Leakage Reduction
                beta_leak = np.linalg.pinv(y)@x
                xc = x - y@beta_leak

                # hilbert
                ht = hilbert(np.concatenate([xc, y], axis=1), axis=0)
                ht = ht[cut_amount:ht.shape[0] - cut_amount, :]
                ht -= np.mean(ht, axis=0)

                env = np.abs(ht)
                corr = pearsonr(env[:, 0], env[:, 1])[0]
                aec[region_i, region_j] = np.abs(corr)

        return aec


class BasePLI(ConnectivityMeasure):

    def __init__(self, window_size, step_size, fmin, fmax, n_surrogates=20, verbose=False):
        super().__init__(window_size, step_size, fmin, fmax, verbose=verbose)
        self.n_surrogates = n_surrogates

    def compute(self, data, fs):
        windows = super().compute(data, fs)
        pli_windows = []
        for n, window in enumerate(windows):
            if self.verbose:
                print(f"Computing on window {n+1}/{len(windows)}")
            pli_window = self._pli_window(window, self._pli)
            if self.n_surrogates > 0:
                pli_window = self._correct_pli_window(window, pli_window)
            pli_windows.append(pli_window)
        pli_windows = np.array(pli_windows)
        return pli_windows

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

    def __init__(self, window_size, step_size, fmin, fmax, n_surrogates=20, verbose=False, p_value=0.05):
        super().__init__(window_size, step_size, fmin, fmax,
                         n_surrogates=n_surrogates, verbose=verbose)
        self.p_value = p_value

    def compute(self, data, fs):
        wpli = super().compute(data, fs)

        # fill up the upper half triangle
        return wpli + wpli.transpose([0, 2, 1])

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

    def __init__(self, window_size, step_size, fmin, fmax, n_surrogates=20, verbose=False, p_value=0.05):
        super().__init__(window_size, step_size, fmin, fmax,
                         n_surrogates=n_surrogates, verbose=verbose)
        self.p_value = p_value

    def compute(self, data, fs):
        dpli = super().compute(data, fs)

        # fill up the upper half triangle
        dpli_final = dpli - np.transpose(dpli, [0, 2, 1])
        i = np.triu_indices(dpli.shape[1], k=1)
        j = np.diag_indices(dpli.shape[1])
        z = np.zeros((1, dpli.shape[1], dpli.shape[1]))
        z[0, i[0], i[1]] = 1
        dpli_final += z
        dpli_final[:, j[0], j[1]] = 0.5
        return dpli_final

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


def connectivity_compute(data, window_size, step_size, fmin, fmax, fs, n_surrogates=0, verbose=True, mode="aec"):

    metrics = {
        "wpli": WPLI,
        "dpli": DPLI,
        "aec": AEC,
    }

    metric_class = metrics[mode]
    if metric_class == AEC:
        assert n_surrogates == 0, "surrogate correction is not supported for the AEC"
        metric_computer = metric_class(
            window_size=window_size, step_size=step_size, fmin=fmin, fmax=fmax, verbose=verbose)
    else:
        metric_computer = metric_class(window_size=window_size, step_size=step_size,
                                       fmin=fmin, fmax=fmax, n_surrogates=n_surrogates, verbose=verbose)

    metric = metric_computer.compute(data, fs)

    return metric
