import numpy as np
from acoustics.bands import octave,third
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.io import wavfile
from scipy.signal import fftconvolve
class FilterBank(object):
    def __init__(self, n, centerFrequencies, samplingRate, maxLength=None):
        self.n = n
        if n % 2 == 0:
            self.n = n + 1
        else:
            self.n = n

        self.centerFrequencies = centerFrequencies
        self.samplingRate = samplingRate
        self.maxLength = maxLength

        centerFrequencies = np.array(centerFrequencies, dtype=np.float)
        centerNormFreqs = centerFrequencies / (self.samplingRate / 2.0)
        cutoffs = centerNormFreqs[:-1] + np.diff(centerNormFreqs) / 2


        filters = []
        for i in range(len(centerFrequencies)):
            if i == 0:
                # Low-pass filter
                b = signal.firwin(self.n, cutoff=cutoffs[0], window='hamming')
            elif i == len(centerFrequencies) - 1:
                # High-pass filter
                b = signal.firwin(
                    self.n, cutoff=cutoffs[-1], window='hamming', pass_zero=False)
            else:
                # Band-pass filter
                b = signal.firwin(
                    self.n, [cutoffs[i - 1], cutoffs[i]], pass_zero=False)

            filters.append(b)
        self.filters = np.array(filters)

        self._precomputeFiltersFourier()

    def _precomputeFiltersFourier(self):
        N = self.filters.shape[-1]
        if self.maxLength is not None:
            N = self.maxLength

        self.filtersFourier = np.fft.fft(self.filters, N)

    def getScaledImpulseResponse(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))
        return np.sum(self.filters * scales[:, np.newaxis], axis=0)

    def getScaledImpulseResponseFourier(self, scales=1):
        if not isinstance(scales, (list, tuple)):
            scales = scales * np.ones(len(self.filters))

        return np.sum(self.filters * scales[:, np.newaxis], axis=0)

    def getScaledImpulseResponse_bandwise(self, signal, index_b):
        return fftconvolve(signal, self.filters[index_b], mode="same")

    def display(self, scales=1, merged=False):
        # Adapted from: http://mpastell.com/2010/01/18/fir-with-scipy/

        if merged:
            b = self.getScaledImpulseResponse(scales)
            filters = [b]
        else:
            filters = np.copy(self.filters)
            if not isinstance(scales, (list, tuple)):
                scales = scales * np.ones(len(filters))
            filters *= scales[:, np.newaxis]

        fig = plt.figure(figsize=(8, 6), facecolor='white', frameon=True)
        for b in filters:
            w, h = signal.freqz(b, 1)
            h_dB = 20 * np.log10(abs(h))
            plt.subplot(211)
            plt.plot(w / max(w), h_dB)
            plt.ylim(-150, 5)
            plt.ylabel('Magnitude (db)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Frequency response')
            plt.subplot(212)
            h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))
            plt.plot(w / max(w), h_Phase)
            plt.ylabel('Phase (radians)')
            plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
            plt.title(r'Phase response')
            plt.subplots_adjust(hspace=0.5)
        return fig
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)

from scipy.io import wavfile
from scipy import stats

from acoustics.utils import _is_1d
from acoustics.signal import bandpass
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)

SOUNDSPEED = 343.0
real_filename ='/home/jzhou3083/work/evaluation/measurement/RIR_LS1_MP3.wav'
simulated_filename= '/home/jzhou3083/work/evaluation/simulation/IR_LS1_MP3.wav'


def t60_impulse(file_name, bands, rt='t30'):  # pylint: disable=too-many-locals
    """
    Reverberation time from a WAV impulse response.

    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`, `'t10'` and `'edt'`.
    :returns: Reverberation time :math:`T_{60}`

    """
    fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
    factor=1

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = bandpass(raw_signal, low[band], high[band], fs, order=8)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch))

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60


def clarity(time, signal, fs, bands=None):
    """
    Clarity :math:`C_i` determined from an impulse response.

    :param time: Time in miliseconds (e.g.: 50, 80).
    :param signal: Impulse response.
    :type signal: :class:`np.ndarray`
    :param fs: Sample frequency.
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`

    """
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    c = np.zeros(bands.size)
    for band in range(bands.size):
        filtered_signal = bandpass(signal, low[band], high[band], fs, order=8)
        h2 = filtered_signal**2.0
        t = int((time / 1000.0) * fs + 1)
        c[band] = 10.0 * np.log10((np.sum(h2[:t]) / np.sum(h2[t:])))
    return c


def c50_from_file(file_name, bands=None):
    """
    Clarity for 50 miliseconds :math:`C_{50}` from a file.

    :param file_name: File name (only WAV is supported).
    :type file_name: :class:`str`
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`

    """
    fs, signal = wavfile.read(file_name)
    return clarity(50.0, signal, fs, bands)


def c80_from_file(file_name, bands=None):
    """
    Clarity for 80 miliseconds :math:`C_{80}` from a file.

    :param file_name: File name (only WAV is supported).
    :type file_name: :class:`str`
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`

    """
    fs, signal = wavfile.read(file_name)
    return clarity(80.0, signal, fs, bands)


def DIC_drom_file(file_name, bands=None):
    """
    """
    fs, signal = wavfile.read(file_name)
    return Dic(50, signal, fs, bands)
def Dic(time, signal, fs, bands=None):
    """
    Clarity :math:`C_i` determined from an impulse response.

    :param time: Time in miliseconds (e.g.: 50, 80).
    :param signal: Impulse response.
    :type signal: :class:`np.ndarray`
    :param fs: Sample frequency.
    :param bands: Bands of calculation (optional). Only support standard octave and third-octave bands.
    :type bands: :class:`np.ndarray`

    """
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    c = np.zeros(bands.size)
    for band in range(bands.size):
        filtered_signal = bandpass(signal, low[band], high[band], fs, order=8)
        h2 = filtered_signal**2.0
        t = int((time / 1000.0) * fs + 1)
        c[band] = (np.sum(h2[:t]) / np.sum(h2))
    return c

nbsrc = 2
nbmic = 5

band = octave(125,8000)
print(band)
filter = FilterBank(n=257,centerFrequencies=band,samplingRate=44100)


measured_t30 =0
simulated_t30 =0
measured_edt = 0
simulated_edt = 0
measured_c80 = 0
simulated_c80 = 0
measured_c50 = 0
simulated_c50 = 0
measured_d50 = 0
simulated_d50 = 0

diff_t30 =0
diff_edt =0
diff_c80 = 0
diff_c50 = 0

for src in range(1,3):
    for mic in range(1,6):

        real_filename ='/home/jzhou3083/work/evaluation/measurement/RIR_LS'+str(src)+'_MP'+str(mic)+'.wav'
        simulated_filename =  '/home/jzhou3083/work/evaluation/simulation/IR_LS'+str(src)+'_MP'+str(mic)+'.wav'

        measured_t30 += t60_impulse(real_filename, bands=band, rt='t30')
        simulated_t30 += t60_impulse(simulated_filename, bands=band, rt='t30')
        # print(measured_t30,simulated_t30)
        # diff_t30 +=np.divide(measured_t30-simulated_t30,measured_t30)

        measured_edt += t60_impulse(real_filename, bands=band, rt='edt')
        simulated_edt += t60_impulse(simulated_filename, bands=band, rt='edt')
        # diff_edt +=np.divide(measured_edt-simulated_edt,measured_edt)

        measured_d50 += DIC_drom_file(real_filename, bands=band)
        simulated_d50 += DIC_drom_file(simulated_filename, bands=band)

        measured_c80 += c80_from_file(real_filename, bands=band)
        simulated_c80 += c80_from_file(simulated_filename, bands=band)
        # diff_c80 += measured_c80-simulated_c80

# ## t30 diff
# print(measured_t30)
# print(simulated_t30)
# measured_t30 /=8
# simulated_t30 /=8
# measured_edt /=8
# simulated_edt /=8
# measured_c80 /=8
# simulated_c80 /=8
# measured_c50 /=8
# simulated_c50 /=8

print(band)
fs,data = wavfile.read(simulated_filename)
# wavfile.write('example.wav',44100,new_data)

# measured_t30 = ac.t60_impulse(real_filename, bands=band, rt='t30')
# simulated_t30 = ac.t60_impulse('example.wav', bands=band, rt='t30')




plt.plot(band,measured_edt/10,label="measured")
plt.plot(band,simulated_edt/10,label="simulated")
plt.xlabel("octave(Hz)")
plt.ylabel("ratio(dB)")
plt.title("C80(%)")
# plt.ylim(1,3)
plt.legend()
plt.show()
# rate = 10
#
# print("T30(s) difference (reference: 5% rel )",diff_t30/rate)

#
# ### EDT diff
#
# # diff_edt = np.divide((measured_edt-simulated_edt),measured_edt)
# print("EDT(%) difference (reference: 5% rel )  ", diff_edt/rate)
#
#
# ### c80 diff
# # print(measured_c80)
# # print(simulated_c80)
# diff_c80 = measured_c80-simulated_c80
# print("C80(dB) difference (reference: 1 dB abs )",diff_c80/rate)

# from scipy.io import wavfile
# fs, signal_r = wavfile.read(real_filename)
# fs, signal_s = wavfile.read(simulated_filename)
# signal_r= np.square(signal_r)
# signal_s= np.square(signal_s)#
# print(signal_r[:50000])
# plt.plot(signal_s[:50000])
# plt.plot(signal_s)
# plt.plot(signal_r,label ='real')
# plt.plot(signal_s,lable='simulated')
# plt.legend()
# plt.show()


# plt.plot(measured_c80,label = "real c80")
# plt.plot(simulated_c80,label="simulated c80")
# plt.legend()
# plt.show()

# measured_t30 = ac.t60_impulse(real_filename,bands=band,rt='t30')
# simulated_t30 = ac.t60_impulse(simulated_filename,bands=band,rt='t30')
# # print(measured_t30,simulated_t30)
#
# measured_edt = ac.t60_impulse(real_filename,bands=band,rt='edt')
# simulated_edt = ac.t60_impulse(simulated_filename,bands=band,rt='edt')
# # print(measured_edt, simulated_edt)
#
#
# measured_c80 = ac.c80_from_file(real_filename,bands =band)
# simulated_c80 = ac.c80_from_file(simulated_filename, bands = band)
# # print(measured_c80,simulated_c80)
#
#
# measured_c50 =ac.c50_from_file(real_filename,bands =band)
# simulated_c50 = ac.c50_from_file(simulated_filename,bands = band)
# print(measured_c50,simulated_c50)



