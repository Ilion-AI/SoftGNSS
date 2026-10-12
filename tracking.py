import numpy as np

from initialize import Result


class TrackingResult(Result):
    def __init__(self, acqResult):
        self._results = None
        self._channels = acqResult.channels
        self._settings = acqResult.settings

    def track(self, fid):
        channel = self._channels
        settings = self._settings

        codePeriods = settings.msToProcess
        earlyLateSpc = settings.dllCorrelatorSpacing
        PDIcode = 0.001
        tau1code, tau2code = settings.calcLoopCoef(
            settings.dllNoiseBandwidth, settings.dllDampingRatio, 1.0)
        PDIcarr = 0.001
        tau1carr, tau2carr = settings.calcLoopCoef(
            settings.pllNoiseBandwidth, settings.pllDampingRatio, 0.25)

        rec = []
        isIQ = getattr(settings, 'isComplexIQ', False)

        for channelNr in range(settings.numberOfChannels):
            msToProcess = int(settings.msToProcess)
            status = '-'

            absoluteSample = np.zeros(msToProcess)
            codeFreq_      = np.inf * np.ones(msToProcess)
            carrFreq_      = np.inf * np.ones(msToProcess)
            I_P_ = np.zeros(msToProcess); I_E_ = np.zeros(msToProcess); I_L_ = np.zeros(msToProcess)
            Q_E_ = np.zeros(msToProcess); Q_P_ = np.zeros(msToProcess); Q_L_ = np.zeros(msToProcess)
            dllDiscr     = np.inf * np.ones(msToProcess)
            dllDiscrFilt = np.inf * np.ones(msToProcess)
            pllDiscr     = np.inf * np.ones(msToProcess)
            pllDiscrFilt = np.inf * np.ones(msToProcess)
            PRN = 0

            if channel[channelNr].PRN != 0:
                PRN = channel[channelNr].PRN

                if isIQ:
                    byte_offset = (settings.skipNumberOfBytes
                                   + int(channel[channelNr].codePhase) * settings.bytesPerSample * 2)
                else:
                    byte_offset = (settings.skipNumberOfBytes
                                   + int(channel[channelNr].codePhase))
                fid.seek(byte_offset, 0)

                caCode = settings.generateCAcode(channel[channelNr].PRN - 1)
                caCode = np.r_[caCode[-1], caCode, caCode[0]]

                codeFreq = settings.codeFreqBasis
                remCodePhase = 0.0
                carrFreq = channel[channelNr].acquiredFreq
                carrFreqBasis = channel[channelNr].acquiredFreq
                remCarrPhase = 0.0
                oldCodeNco = 0.0;  oldCodeError = 0.0
                oldCarrNco = 0.0;  oldCarrError = 0.0

                for loopCnt in range(msToProcess):
                    if loopCnt % 50 == 0:
                        try:
                            print('Tracking: Ch %d' % (channelNr + 1) +
                                  ' of %d' % settings.numberOfChannels +
                                  '; PRN#%02d' % channel[channelNr].PRN +
                                  '; Completed %d' % loopCnt +
                                  ' of %d' % codePeriods + ' msec')
                        finally:
                            pass

                    codePhaseStep = codeFreq / settings.samplingFreq
                    blksize = int(np.ceil((settings.codeLength - remCodePhase) / codePhaseStep))

                    if isIQ:
                        raw = np.fromfile(fid, settings.dataType, blksize * 2)
                        if len(raw) < 2:
                            print('Not enough samples for tracking, exiting!')
                            fid.close()
                            return None
                        raw = raw[: (len(raw) // 2) * 2]
                        rawSignal = (raw[0::2].astype(np.float64)
                                     + 1j * raw[1::2].astype(np.float64))
                        blksize = len(rawSignal)
                    else:
                        rawSignal = np.fromfile(fid, settings.dataType, blksize)
                        if len(rawSignal) != blksize:
                            print('Not enough samples for tracking, exiting!')
                            fid.close()
                            return None

                    tcode_e = np.linspace(
                        remCodePhase - earlyLateSpc,
                        blksize * codePhaseStep + remCodePhase - earlyLateSpc,
                        blksize, endpoint=False)
                    earlyCode = caCode[np.ceil(tcode_e).astype(np.int64)]

                    tcode_l = np.linspace(
                        remCodePhase + earlyLateSpc,
                        blksize * codePhaseStep + remCodePhase + earlyLateSpc,
                        blksize, endpoint=False)
                    lateCode = caCode[np.ceil(tcode_l).astype(np.int64)]

                    tcode_p = np.linspace(
                        remCodePhase,
                        blksize * codePhaseStep + remCodePhase,
                        blksize, endpoint=False)
                    promptCode = caCode[np.ceil(tcode_p).astype(np.int64)]

                    remCodePhase = tcode_p[-1] + codePhaseStep - 1023.0

                    time = np.arange(0, blksize + 1) / settings.samplingFreq
                    trigarg = carrFreq * 2.0 * np.pi * time + remCarrPhase
                    remCarrPhase = trigarg[blksize] % (2 * np.pi)

                    if isIQ:
                        baseband = rawSignal * np.exp(-1j * trigarg[:blksize])
                        iBaseband = baseband.real
                        qBaseband = baseband.imag
                    else:
                        carrCos = np.cos(trigarg[:blksize])
                        carrSin = np.sin(trigarg[:blksize])
                        qBaseband = carrCos * rawSignal
                        iBaseband = carrSin * rawSignal

                    I_E = (earlyCode  * iBaseband).sum()
                    Q_E = (earlyCode  * qBaseband).sum()
                    I_P = (promptCode * iBaseband).sum()
                    Q_P = (promptCode * qBaseband).sum()
                    I_L = (lateCode   * iBaseband).sum()
                    Q_L = (lateCode   * qBaseband).sum()

                    carrError = np.arctan(Q_P / I_P) / 2.0 / np.pi
                    carrNco = (oldCarrNco
                               + tau2carr / tau1carr * (carrError - oldCarrError)
                               + carrError * (PDIcarr / tau1carr))
                    oldCarrNco = carrNco
                    oldCarrError = carrError
                    carrFreq = carrFreqBasis + carrNco
                    carrFreq_[loopCnt] = carrFreq

                    e_mag = np.sqrt(I_E ** 2 + Q_E ** 2)
                    l_mag = np.sqrt(I_L ** 2 + Q_L ** 2)
                    denom = e_mag + l_mag
                    codeError = (e_mag - l_mag) / denom if denom > 0 else 0.0

                    codeNco = (oldCodeNco
                               + tau2code / tau1code * (codeError - oldCodeError)
                               + codeError * (PDIcode / tau1code))
                    oldCodeNco = codeNco
                    oldCodeError = codeError
                    codeFreq = settings.codeFreqBasis - codeNco
                    codeFreq_[loopCnt] = codeFreq

                    absoluteSample[loopCnt] = fid.tell()
                    dllDiscr[loopCnt]     = codeError
                    dllDiscrFilt[loopCnt] = codeNco
                    pllDiscr[loopCnt]     = carrError
                    pllDiscrFilt[loopCnt] = carrNco
                    I_E_[loopCnt] = I_E;  I_P_[loopCnt] = I_P;  I_L_[loopCnt] = I_L
                    Q_E_[loopCnt] = Q_E;  Q_P_[loopCnt] = Q_P;  Q_L_[loopCnt] = Q_L

                status = channel[channelNr].status
                rec.append((status, absoluteSample, codeFreq_, carrFreq_,
                            I_P_, I_E_, I_L_, Q_E_, Q_P_, Q_L_,
                            dllDiscr, dllDiscrFilt, pllDiscr, pllDiscrFilt, PRN))

        trackResults = np.rec.fromrecords(rec,
            dtype=[('status', 'S1'), ('absoluteSample', 'object'),
                   ('codeFreq', 'object'), ('carrFreq', 'object'),
                   ('I_P', 'object'), ('I_E', 'object'), ('I_L', 'object'),
                   ('Q_E', 'object'), ('Q_P', 'object'), ('Q_L', 'object'),
                   ('dllDiscr', 'object'), ('dllDiscrFilt', 'object'),
                   ('pllDiscr', 'object'), ('pllDiscrFilt', 'object'),
                   ('PRN', 'int64')])
        self._results = trackResults
        return

    def plot(self):
        import matplotlib as mpl
        import matplotlib.gridspec as gs
        import matplotlib.pyplot as plt

        mpl.rcdefaults()
        mpl.rc('savefig', bbox='tight', transparent=False, format='png')
        mpl.rc('axes', grid=True, linewidth=1.5, axisbelow=True)
        mpl.rc('lines', linewidth=1.5, solid_joinstyle='bevel')
        mpl.rc('figure', figsize=[8, 6], dpi=120)
        mpl.rc('text', usetex=True)
        mpl.rc('font', family='serif', serif='Computer Modern Roman', size=8)
        mpl.rc('mathtext', fontset='cm')

        trackResults = self._results
        settings = self._settings
        channelList = np.intersect1d(range(settings.numberOfChannels),
                                     range(len(trackResults)))

        for channelNr in channelList:
            f = plt.figure(channelNr + 200)
            f.set_label('Channel %d (PRN %d) results' % (channelNr, trackResults[channelNr].PRN))
            spec = gs.GridSpec(3, 3)

            h11 = plt.subplot(spec[0, 0])
            h12 = plt.subplot(spec[0, 1:])
            h21 = plt.subplot(spec[1, 0])
            h22 = plt.subplot(spec[1, 1:])
            h31 = plt.subplot(spec[2, 0])
            h32 = plt.subplot(spec[2, 1])
            h33 = plt.subplot(spec[2, 2])

            t = np.arange(settings.msToProcess) / 1000.0
            tr = trackResults[channelNr]

            h11.plot(tr.I_P, tr.Q_P, '.')
            h11.set(title='I/Q Scatter', xlabel='I prompt', ylabel='Q prompt')
            h11.grid(); h11.axis('equal')

            h12.plot(t, tr.I_P)
            h12.set(title='Navigation bits', xlabel='Time (s)')
            h12.grid(); h12.axis('tight')

            h21.plot(t, tr.pllDiscr, 'r')
            h21.set(xlabel='Time (s)', ylabel='Amplitude', title='Raw PLL')
            h21.grid(); h21.axis('tight')

            h22.plot(t, np.sqrt(tr.I_E**2 + tr.Q_E**2).T,
                     t, np.sqrt(tr.I_P**2 + tr.Q_P**2).T,
                     t, np.sqrt(tr.I_L**2 + tr.Q_L**2).T, '-*')
            h22.set(title='Correlation', xlabel='Time (s)')
            h22.legend(['Early', 'Prompt', 'Late'])
            h22.grid(); h22.axis('tight')

            h31.plot(t, tr.pllDiscrFilt, 'b')
            h31.set(xlabel='Time (s)', ylabel='Amplitude', title='Filtered PLL')
            h31.grid(); h31.axis('tight')

            h32.plot(t, tr.dllDiscr, 'r')
            h32.set(xlabel='Time (s)', ylabel='Amplitude', title='Raw DLL')
            h32.grid(); h32.axis('tight')

            h33.plot(t, tr.dllDiscrFilt, 'b')
            h33.set(xlabel='Time (s)', ylabel='Amplitude', title='Filtered DLL')
            h33.grid(); h33.axis('tight')

            f.show()