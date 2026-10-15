import numpy as np

from initialize import Result


class AcquisitionResult(Result):
    def __init__(self, settings):
        self._settings = settings
        self._results = None
        self._channels = None

    @property
    def peakMetric(self):
        assert isinstance(self._results, np.recarray)
        return self._results.peakMetric

    @property
    def carrFreq(self):
        assert isinstance(self._results, np.recarray)
        return self._results.carrFreq

    @property
    def codePhase(self):
        assert isinstance(self._results, np.recarray)
        return self._results.codePhase

    def acquire(self, longSignal):
        """Parallel Code Phase Search acquisition.

        Works with complex baseband I/Q (isComplexIQ=True, IF=0)
        and legacy real-valued IF signals (isComplexIQ=False).
        """
        settings = self._settings
        samplesPerCode = settings.samplesPerCode
        ts = 1.0 / settings.samplingFreq
        isIQ = getattr(settings, 'isComplexIQ', False)

        numberOfFrqBins = int(np.round(settings.acqSearchBand * 2) + 1)
        caCodesTable = settings.makeCaTable()

        results = np.zeros((numberOfFrqBins, samplesPerCode))
        frqBins = np.zeros(numberOfFrqBins)
        carrFreq   = np.zeros(32)
        codePhase_ = np.zeros(32)
        peakMetric = np.zeros(32)

        if isIQ:
            signal0DC = longSignal - longSignal.mean()
            sig1 = signal0DC[0:samplesPerCode]
            sig2 = signal0DC[samplesPerCode:2 * samplesPerCode]
            t_one_ms = np.arange(samplesPerCode) * ts
        else:
            sig1 = longSignal[0:samplesPerCode]
            sig2 = longSignal[samplesPerCode:2 * samplesPerCode]
            signal0DC = longSignal - longSignal.mean()
            phasePoints = np.arange(samplesPerCode) * 2 * np.pi * ts

        samplesPerCodeChip = int(round(settings.samplingFreq / settings.codeFreqBasis))

        print('(')
        for PRN in range(len(settings.acqSatelliteList)):
            caCodeFreqDom = np.fft.fft(caCodesTable[PRN, :]).conj()

            for frqBinIndex in range(numberOfFrqBins):
                frqBins[frqBinIndex] = (settings.IF
                                        - settings.acqSearchBand / 2 * 1000
                                        + 500.0 * frqBinIndex)
                fd = frqBins[frqBinIndex]

                if isIQ:
                    carrier = np.exp(-1j * 2 * np.pi * fd * t_one_ms)
                    bb1 = sig1 * carrier
                    bb2 = sig2 * carrier
                    acqRes1 = np.abs(np.fft.ifft(np.fft.fft(bb1) * caCodeFreqDom)) ** 2
                    acqRes2 = np.abs(np.fft.ifft(np.fft.fft(bb2) * caCodeFreqDom)) ** 2
                else:
                    sinCarr = np.sin(fd * phasePoints)
                    cosCarr = np.cos(fd * phasePoints)
                    IQ1 = sinCarr * sig1 + 1j * cosCarr * sig1
                    IQ2 = sinCarr * sig2 + 1j * cosCarr * sig2
                    acqRes1 = np.abs(np.fft.ifft(np.fft.fft(IQ1) * caCodeFreqDom)) ** 2
                    acqRes2 = np.abs(np.fft.ifft(np.fft.fft(IQ2) * caCodeFreqDom)) ** 2

                if acqRes1.max() > acqRes2.max():
                    results[frqBinIndex, :] = acqRes1
                else:
                    results[frqBinIndex, :] = acqRes2

            # SNR-based detection
            peak_idx = np.unravel_index(results.argmax(), results.shape)
            peak_fd_bin = peak_idx[0]
            codePhase = peak_idx[1]
            peakSize = results[peak_fd_bin, codePhase]

            mask = np.ones(samplesPerCode, dtype=bool)
            excl1 = codePhase - samplesPerCodeChip
            excl2 = codePhase + samplesPerCodeChip
            if excl1 <= 0:
                mask[excl2:samplesPerCode + excl1 + 1] = False
            elif excl2 >= samplesPerCode - 1:
                mask[excl2 - samplesPerCode:excl1] = False
            else:
                mask[:excl1 + 1] = False
                mask[excl2:] = False

            noise_vals = results[peak_fd_bin, mask]
            noise_mean = noise_vals.mean()
            noise_std  = noise_vals.std()
            snr = (peakSize - noise_mean) / noise_std if noise_std > 0 else 0.0
            peakMetric[PRN] = snr

            if snr > settings.acqThreshold:
                print('%02d ' % (PRN + 1))
                if isIQ:
                    carrFreq[PRN] = frqBins[peak_fd_bin]
                else:
                    caCode = settings.generateCAcode(PRN)
                    codeValueIndex = np.floor(
                        ts * np.arange(1, 10 * samplesPerCode + 1)
                        / (1.0 / settings.codeFreqBasis)
                    ).astype(np.int64)
                    longCaCode = caCode[codeValueIndex % 1023]
                    xCarrier = (signal0DC[codePhase:codePhase + 10 * samplesPerCode]
                                * longCaCode)
                    fftNumPts = int(8 * 2 ** np.ceil(np.log2(len(xCarrier))))
                    fftxc = np.abs(np.fft.fft(xCarrier, fftNumPts))
                    uniqFftPts = int(np.ceil((fftNumPts + 1) / 2.0))
                    fftMaxIndex = fftxc[4:uniqFftPts - 5].argmax()
                    fftFreqBins = np.arange(uniqFftPts) * settings.samplingFreq / fftNumPts
                    carrFreq[PRN] = fftFreqBins[fftMaxIndex]
                codePhase_[PRN] = codePhase
            else:
                print('. ')

        print(')\n')
        acqResults = np.core.records.fromarrays(
            [carrFreq, codePhase_, peakMetric],
            names='carrFreq,codePhase,peakMetric')
        self._results = acqResults
        return

    def plot(self):
        assert isinstance(self._results, np.recarray)
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcdefaults()
        mpl.rc('savefig', bbox='tight', transparent=False, format='png')
        mpl.rc('axes', grid=True, linewidth=1.5, axisbelow=True)
        mpl.rc('lines', linewidth=1.5, solid_joinstyle='bevel')
        mpl.rc('figure', figsize=[8, 6], autolayout=False, dpi=120)
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif', serif='Computer Modern Roman', size=16)
        mpl.rc('mathtext', fontset='cm')

        f, hAxes = plt.subplots()
        plt.bar(range(1, 33), self.peakMetric)
        plt.title('Acquisition results')
        plt.xlabel('PRN number')
        plt.ylabel('Acquisition Metric (SNR)')
        oldAxis = plt.axis()
        plt.axis([0, 33, 0, oldAxis[-1]])
        plt.xticks(range(1, 33), size=12)
        hAxes.xaxis.grid()
        acquiredSignals = self.peakMetric * (self.carrFreq > 0)
        plt.bar(range(1, 33), acquiredSignals, FaceColor=(0, 0.8, 0))
        plt.legend(['Not acquired signals', 'Acquired signals'])
        plt.show()

    def preRun(self):
        assert isinstance(self._results, np.recarray)
        settings = self._settings

        PRN = np.zeros(settings.numberOfChannels, dtype='int64')
        acquiredFreq = np.zeros(settings.numberOfChannels)
        codePhase = np.zeros(settings.numberOfChannels)
        status = ['-' for _ in range(settings.numberOfChannels)]

        PRNindexes = sorted(enumerate(self.peakMetric),
                            key=lambda x: x[-1], reverse=True)

        ii = 0
        for idx, _ in PRNindexes:
            if ii >= settings.numberOfChannels:
                break
            if self.carrFreq[idx] > 0:
                PRN[ii] = idx + 1
                acquiredFreq[ii] = self.carrFreq[idx]
                codePhase[ii] = self.codePhase[idx]
                status[ii] = 'T'
                ii += 1

        channel = np.core.records.fromarrays(
            [PRN, acquiredFreq, codePhase, status],
            names='PRN,acquiredFreq,codePhase,status')
        self._channels = channel
        return

    def showChannelStatus(self):
        channel = self._channels
        settings = self._settings
        assert isinstance(channel, np.recarray)
        print('\n*=========*=====*===============*===========*=============*========*')
        print('| Channel | PRN |   Frequency   |  Doppler  | Code Offset | Status |')
        print('*=========*=====*===============*===========*=============*========*')
        for channelNr in range(settings.numberOfChannels):
            if channel[channelNr].status != '-':
                print('|      %2d | %3d |  %2.5e |   %5.0f   |    %6d   |     %1s  |' % (
                    channelNr,
                    channel[channelNr].PRN,
                    channel[channelNr].acquiredFreq,
                    channel[channelNr].acquiredFreq - settings.IF,
                    channel[channelNr].codePhase,
                    channel[channelNr].status))
            else:
                print('|      %2d | --- |  ------------ |   -----   |    ------   |   Off  |' % channelNr)
        print('*=========*=====*===============*===========*=============*========*\n')


if __name__ == '__main__':
    pass