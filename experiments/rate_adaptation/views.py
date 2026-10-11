"""Signal views retain duration, sensor placement, and a known sampling rate."""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def rate_view(x, rate, anti_alias=True):
    if rate == 30:
        return x.copy()
    if not 0 < rate <= 30:
        raise ValueError("rate must lie in (0, 30]")
    ratio = Fraction(int(rate), 30)
    sample_grid = np.arange(int(np.ceil(x.shape[1] * rate / 30))) * 30 / rate
    if anti_alias:
        low = resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
    else:
        if ratio.numerator != 1:
            raise ValueError("naive decimation control requires an integer divisor of 30 Hz")
        low = x[:, ::ratio.denominator, :]
    high_grid = np.arange(x.shape[1])
    result = np.empty_like(x)
    for i in range(len(x)):
        for channel in range(x.shape[2]):
            result[i, :, channel] = np.interp(high_grid, sample_grid, low[i, :, channel])
    return result


def shared_band(x, rate, guard=0.8):
    """Use a guard band below the lower-rate Nyquist frequency during training."""
    cutoff = guard * rate / 2
    spectrum = np.fft.rfft(x, axis=1)
    frequencies = np.fft.rfftfreq(x.shape[1], d=1 / 30)
    spectrum[:, frequencies > cutoff, :] = 0
    return np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(x.dtype)
