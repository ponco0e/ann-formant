# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:05:56 2015

@author: ponco
"""

import neurolab as nl
from features import mfcc
from features import logfbank
import praatUtil
import matplotlibUtil
import os
import scipy.io.wavfile as wav
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from recorder import *
import math
from scipy.signal import lfilter
from scikits.talkbox import lpc

path = "/home/ponco/devel/mel_cepstral_coeff_neural/vowels/"

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 25), ylim=(-84, 80))
#ax = plt.axes(xlim=(0, 25), ylim=(0, 20))
line, = ax.plot([], [], lw=2)

#MEL
(rate,sig) = wav.read("Ah.wav")
mfcc_feat = mfcc(sig,rate,numcep=30,appendEnergy=False)
fbank_feat = logfbank(sig,rate,nfilt=40)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    #x = np.linspace(0, 12,13)
    x = np.linspace(0, 25,26)
    y = mfcc_feat[i,:]
    #y = fbank_feat[i,:]
    #print("x:" , x.shape)
    #print("y:" , y.shape)
    
    #y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

def get_formants(x):

    # Read from file.
    #spf = wave.open(file_path, 'r') # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

    # Get file as numpy array.
    #x = spf.readframes(-1)
    #x = np.fromstring(x, 'Int16')

    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    
    #Fs = spf.getframerate()
    Fs = 44100
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

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=219, interval=50, blit=True)


#f1=np.argsort(fbsum)[-1]
#f2=np.argsort(fbsum)[-2]
dictForm = { "A":{"f1":[] , "f2":[] }  , "E":{"f1":[] , "f2":[] } , "I":{"f1":[] , "f2":[] } , "O":{"f1":[] , "f2":[] } , "U":{"f1":[] , "f2":[] } } #diccionario con formantes de cada
for wavFile in os.listdir(path):
    if wavFile[-4:]== ".wav" : #si es wav
        print "Formantes de " + wavFile
        formant=praatUtil.calculateFormants(path + wavFile)[0]
        
        for frame in np.arange(50,formant.getNumFrames(),20):
            formantList = formant.get(frame)[1]
            #dictForm[wavFile[0]].append( [ formantList[0]["frequency"] , formantList[1]["frequency"] ] ) #f1 y f2 a la vocal que pertenezcan
            dictForm[wavFile[0]]["f1"].append(formantList[0]["frequency"])
            dictForm[wavFile[0]]["f2"].append(formantList[1]["frequency"])


graph = matplotlibUtil.CGraph(width = 6, height = 6)
graph.createFigure()
ax = graph.getArrAx()[0]
for vowel in dictForm:
	print vowel, len(dictForm[vowel]['f1'])
	ax.plot(dictForm[vowel]['f1'], dictForm[vowel]['f2'], 'o', \
		markersize = 5, alpha = 0.4, label=vowel)
ax.grid()
ax.set_xlabel("F1 [Hz]")
ax.set_ylabel("F2 [Hz]")
ax.set_title("F1/F2 plot")
plt.legend(loc=0)
graph.padding = 0.1
graph.adjustPadding(left = 1.5)
#plt.savefig('formantMel.png')
        
nns = [ nl.net.newff([[1, 1600], [1, 3000]], [5, 1]) for x in xrange(5)] #arreglo de redes neuronales

stimA=np.array((dictForm["A"]["f1"],dictForm["A"]["f2"])).T
stimE=np.array((dictForm["E"]["f1"],dictForm["E"]["f2"])).T
stimI=np.array((dictForm["I"]["f1"],dictForm["I"]["f2"])).T
stimO=np.array((dictForm["O"]["f1"],dictForm["O"]["f2"])).T
stimU=np.array((dictForm["U"]["f1"],dictForm["U"]["f2"])).T
stimArr=[ len(x) for x in [stimA,stimE,stimI,stimO,stimU] ]
stim=np.concatenate((stimA,stimE,stimI,stimO,stimU))

for numNet in xrange(5):
    numOnes = stimArr[numNet]
    if stimArr[:numNet]:
        numBefore = reduce( lambda x,y:x+y,stimArr[:numNet] ) 
    else: numBefore = 0
    if stimArr[numNet+1:]:
        numAfter=reduce( lambda x,y:x+y,stimArr[numNet+1:] )
    else: numAfter=0
    target = np.concatenate((np.zeros(numBefore),np.ones(numOnes),np.zeros(numAfter)))
    target=target.reshape(len(target),1)
    
    print "Entrenando red neuronal " + str(numNet)
    err = nns[numNet].train(stim, target, show=15)
    
    
#target=np.zeros( stim.shape[0] )
#target=target.reshape(len(target),1)


print "Entrenamiento finalizado"

result = nns[0].sim([[850, 1610]]) #formantes de A segun wikipedia
print "A="
print result

result = nns[1].sim([[610, 1900]]) #formantes de E segun wikipedia
print "E="
print result

result = nns[2].sim([[240, 2400]]) #formantes de I segun wikipedia
print "I="
print result

result = nns[3].sim([[360, 640]]) #formantes de O segun wikipedia
print "O="
print result

result = nns[4].sim([[250, 595]]) #formantes de U segun wikipedia
print "U="
print result

plt.show()

fbsum=fbank_feat.sum(axis=0)
plt.plot(fbsum)

SR=SwhRecorder()
SR.setup()

vowels=["A","E","I","O","U"]
while True:
    maxProb=0
    favVowel=0
    f1,f2 = get_formants(SR.getAudio())[1:3]
    for i in xrange(5):
        result = nns[i].sim([[f1, f2]])
        favVowel = i if result > maxProb else favVowel
        maxProb = result if result > maxProb else maxProb
    print vowels[favVowel]