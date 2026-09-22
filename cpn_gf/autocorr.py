import numpy as np

def autocorr_fft(x):
    """
    Autocorrelation of a (possibly complex) 1D array using full FFT.
    Returns ACF for lags 0..N-1, normalized so that ACF[0] = 1.
    """
    x = np.asarray(x)
    n = x.size
    x = x - x.mean()
    
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n]
    acf = acf.real                      # autocorrelation is real even for complex x
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time for real or complex arrays.
    If x is multidimensional, compute along the last axis.
    """
    x = np.asarray(x)
    
    if x.ndim > 1:
        return np.apply_along_axis(integrated_autocorr_time, -1, x, c)

    ac = autocorr_fft(x)
    
    tau = 0.5
    for t in range(1, len(ac)):
        tau += ac[t]
        if t > c * tau:
            break
    return tau