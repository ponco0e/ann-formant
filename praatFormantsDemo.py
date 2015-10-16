# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:05:56 2015

@author: ponco
"""
import praatUtil
import matplotlibUtil
import generalUtility
import praatTextGrid
from matplotlib import pyplot as plt


fName = 'AEIOU_vocalFry.wav'

# in order for this to work, you need to specify a path. change as appropriate
path = "/home/ponco/Documentos/devel/mel_cepstral_coeff_neural/"

# assemble a Praat script to analyze the formants of the file. Make sure that 
# you add a backslach in front of every single quote of your Praat script (i.e.,
# every ' turns into /' - this does not apply here, since the Praat script below 
# does not contain any single quotes). Also, add a new line (backslash n) at the
# end of every line in the script
#
# In particular, we'll create the script below. Note how the path and file names
# are being replaced by variables, so that you can easily change them.
# 
# do ("Read from file...", "/Users/ch/data/programming/python/lib/demo/AEIOU_vocalFry.wav")
# do ("To Formant (burg)...", 0, 5, 5000, 0.025, 50)
# do ("Save as short text file...", "/Users/ch/data/programming/python/lib/demo/AEIOU_vocalFry.Formant")
fileNameOnly = generalUtility.getFileNameOnly(fName)
script = ''
script += 'do ("Read from file...", "' +  path + fName + '")\n'
script += 'do ("To Formant (burg)...", 0, 5, 5000, 0.025, 50)\n'
script += 'do ("Save as short text file...", "' + path + fileNameOnly \
	+ '.Formant")\n'
elapsed = praatUtil.runPraatScript(script)
print "Praat script executed in " + str(elapsed) + " seconds."


# read the generated Praat formants file
formants = praatUtil.PraatFormants()
formants.readFile(fileNameOnly + '.Formant')		

# read the accompanying Praat text grid (see the Praat TextGrid example for an
# extended documentation). We expect a TextGrid that contains one IntervalTier
# lablled 'vowels'. Within this IntervalTier, the occurring vowels are indicated
textGrid = praatTextGrid.PraatTextGrid(0, 0)
textGridFileName = fileNameOnly + '.TextGrid'
arrTiers = textGrid.readFromFile(textGridFileName)
numTiers = len(arrTiers)
if numTiers != 1:
	raise Exception("we expect exactly one Tier in this file")
tier = arrTiers[0]
if tier.getName() != 'vowels':
	raise Exception("unexpected tier")
	
# parse the TextGrid: create a dictionary that stores a list of start and end
# times of all intervals where that particular vowel occurs (that way we'll
# cater for multiple occurrances of the same vowel in a file, should that ever 
# happen)
arrVowels = {}
for i in range(tier.getSize()):
	if tier.getLabel(i) != '':
		interval = tier.get(i)
		vowel = interval[2]
		if not vowel in arrVowels: 
			arrVowels[vowel] = []
		tStart, tEnd = interval[0], interval[1]
		arrVowels[vowel].append([tStart, tEnd]) 

# analyze the formant data: assign formant data to occurring (annotated) vowels
# where applicable, and discard the other formant data (i.e., that data that 
# occurs in time when no vowel annotation was made)
n = formants.getNumFrames()
arrFormants = {}
arrGraphData = {}
for i in range(n):
	t, formantData = formants.get(i)
	
	# loop over all vowels and all intervals for each vowel
	for vowel in arrVowels:
		for tStart, tEnd in arrVowels[vowel]:
			if t >= tStart and t <= tEnd:
			
				# now we know that that particular formant data chunk is within
				# the interval of a particular vowel annotation. use the formant
				# data in the graph to be generated
				
				# make sure we can actually store the formant data: create a 
				# dictionary holding two lists: one for the first and one for
				# the second formant
				if not vowel in arrGraphData:
					arrGraphData[vowel] = {'f1':[], 'f2':[]}
					
				# only consider 1st and 2nd formant
				arrGraphData[vowel]['f1'].append(formantData[0]['frequency'])
				arrGraphData[vowel]['f2'].append(formantData[1]['frequency'])
				

# finally, generate the graph. We're making use of matplotlib's colour cycle by
# only issuing one plot command per vowel. That way we won't have to deal with
# indicating colours ourselves, making our code flexible so it can deal with any
# number of occurring vowels
graph = matplotlibUtil.CGraph(width = 6, height = 6)
graph.createFigure()
ax = graph.getArrAx()[0]
for vowel in arrGraphData:
	print vowel, len(arrGraphData[vowel]['f1'])
	ax.plot(arrGraphData[vowel]['f1'], arrGraphData[vowel]['f2'], 'o', \
		markersize = 5, alpha = 0.4, label=vowel)
ax.grid()
ax.set_xlabel("F1 [Hz]")
ax.set_ylabel("F2 [Hz]")
ax.set_title("F1/F2 plot")
plt.legend(loc=0)
graph.padding = 0.1
graph.adjustPadding(left = 1.5)
plt.savefig('formantDemo.png')

