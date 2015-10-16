# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 01:28:14 2015

@author: ponco
"""

import numpy as np
import wave
import math
from scipy.signal import lfilter
from scikits.talkbox import lpc
from recorder import *


def get_formants(dummy):

    # Read from file.
    spf = wave.open("/home/ponco/devel/mel_cepstral_coeff_neural/vowels/EMartin.wav", 'r') # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

    # Get file as numpy array.
    x = spf.readframes(-1)
    x = np.fromstring(x, 'Int16')

    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    
    Fs = spf.getframerate()
    #Fs = 44100
    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(x1, ncoeff)

    # Get LPC.
    #A, e, k = lpc(x1, 8)


    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    #Fs = spf.getframerate()
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs

SR=SwhRecorder()
SR.setup()
while True:
    print get_formants(SR.getAudio())[:6]