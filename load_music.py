import sys,os

'''
Load music from converted midi files (stored in the folder "converted").
Function "loadMusic" is called by run.py as soon as you call "from run import *".
'''

class Track:
	def __init__(self):
		self.pitch=[]
		self.dur=[]
		self.resolution=0

class Song:
	def __init__(self):
		self.track=[]

'''
The following 3 functions are called from bottom to top.
'''

def loadTrack(path):
	file=open(path)
	lines=file.readlines()
	track=Track()
	track.resolution=int(lines[0])
	for i in xrange(1,len(lines)):
		data=lines[i].split(' ')
		track.pitch.append(int(data[0]))
		track.dur.append(float(data[1]))
	return track

def loadSong(path):
	track_namelist=os.listdir(path)
	ret=Song()
	for trackname in track_namelist:
		track=loadTrack(path+'\\'+trackname)
		ret.track.append(track)
	return ret

def loadMusic(path):
	namelist=os.listdir(path)
	ret=[]
	for musicname in namelist:
		song=loadSong(path+'\\'+musicname)
		ret.append(song)
	return ret