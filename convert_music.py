import os,sys
import midi

pitchname=['A','#A','B','C','#C','D','#D','E','F','#F','G','#G']
def toNote(n):
	return pitchname[(n-21)%12]


'''
This python script converts every midi file in ./midi to converted format.
Before training the network, make sure you run "python convert_music.py".
'''

'''
This flag controls whether the top track of a midi is the only one used.
For example, for a Bach piano piece, both two melodies are equally important.
For a Chopin waltz, only the right hand plays the melody, the left hand plays accompaniment.
'''
topTrackOnly=True


path=sys.path[0]+'\\midi\\'
txtpath=sys.path[0]+'\\converted\\'
midilist=os.listdir(path)
for filename in midilist:
	pattern=midi.read_midifile(path+filename)
	print(pattern)
	trackid=0
	for track in pattern:
		i=0
		while i<len(track) and not isinstance(track[i],midi.NoteOnEvent):
			i+=1
		if i>=len(track):
			continue
		if not os.path.exists(txtpath+filename[0:-4]):
			os.makedirs(txtpath+filename[0:-4])
		fp=open(txtpath+filename[0:-4]+'\\'+filename[0:-4]+'_'+str(trackid)+'.txt',"w")
		fp.write(str(pattern.resolution)+'\n')
		pitch=[]
		beg=[]
		end=[]
		cur=-1
		time=0
		while i<len(track):
			e=track[i]
			if isinstance(e,midi.EndOfTrackEvent):
				break
			time+=e.tick
			if isinstance(e,midi.NoteOnEvent) and e.data[1]>0:
				pitch.append(e.data[0])
				beg.append(time)
				end.append(-1)
				cur+=1
			if isinstance(e,midi.NoteOffEvent) or (isinstance(e,midi.NoteOnEvent) and e.data[1]==0):
				j=cur
				while j>=0:
					if pitch[j]==e.data[0]:
						break
					j-=1
				end[j]=time
			i+=1
		i=0
		lastend=0
		while i<len(pitch):
			j=i
			while j<len(pitch) and beg[j]==beg[i]: j+=1
			if beg[i]>lastend:
				fp.write('0 '+str(beg[i]-lastend)+'\n')
			toppitch=max(pitch[i:j])
			duration=end[i]-beg[i]
			lastend=end[i]
			fp.write(str(toppitch)+' '+str(duration)+'\n')
			if duration<0:
				fp.close()
				os.remove(txtpath+filename[0:-4]+'\\'+filename[0:-4]+'_'+str(trackid)+'.txt')
				break
			i=j
		trackid+=1
		if topTrackOnly: break