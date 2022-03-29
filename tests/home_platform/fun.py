import math

import matplotlib.pyplot as plt
import numpy as np
def fast_rir_builder( rir, time, alpha, fs, fdl=81, lut_gran=20):

    '''
    Fast impulse response builder. This function takes the image source delays
    and amplitude and fills the impulse response. It uses a linear interpolation
    of the sinc function for speed.

    Parameters
    ----------
    rir: ndarray (double)
        The array to receive the impulse response. It should be filled with
        zero and of the correct size
    time: ndarray (double)
        The array of delays
    alpha: ndarray (double)
        The array of attenuations
    fs: int
        The sampling frequency
    fdl: int
        The length of the fractional delay filter (should be odd)
    lut_gran: int
        The number of point per unit in the sinc interpolation table
    '''

    fdl2 = (fdl - 1) // 2
    n_times = time.shape[0]
    assert time.shape[0] == alpha.shape[0]
    assert fdl % 2 == 1

    # check the size of the return array
    max_sample = math.ceil(fs * np.max(time)) + fdl2
    min_sample = math.floor(fs * np.min(time)) - fdl2
    assert min_sample >= 0
    assert max_sample < rir.shape[0]

    # create a look-up table of the sinc function and
    # then use linear interpolation
    delta = 1. / lut_gran
    lut_size = (fdl + 1) * lut_gran + 1
    n = np.linspace(-fdl2-1, fdl2 + 1, lut_size)

    sinc_lut = np.sinc(n)
    hann = np.hanning(fdl)


    for i in range(n_times):
        # decompose integer and fractional delay
        sample_frac = fs * time[i]
        time_ip = int(math.floor(sample_frac))
        time_fp = sample_frac - time_ip

        # do the linear interpolation
        x_off_frac = (1. - time_fp) * lut_gran
        lut_gran_off = int(math.floor(x_off_frac))
        x_off = (x_off_frac - lut_gran_off)
        lut_pos = lut_gran_off
        k = 0
        for f in range(-fdl2, fdl2+1):
            rir[time_ip + f] += alpha[i] * hann[k] * (sinc_lut[lut_pos]
                        + x_off * (sinc_lut[lut_pos+1] - sinc_lut[lut_pos]))
            lut_pos += lut_gran
            k += 1

rir=np.zeros(118001)
fast_rir_builder(rir=rir,time=np.array([0.00742493,0.008]),alpha=np.array([0.44649599,0.9]),fs=44100)
plt.plot(rir[200:400])
plt.show()
