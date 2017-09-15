import numpy as np
import os,random

def decide(note,posi):
	'''
	A function involved in the random process of generating note sequence.
	It picks a candidate out of n candidates by their probability.
	i.e. P(A3)=0.8,P(G3)=0.2
	then the probability to pick A3 is 80%.
	'''

	posi= posi/np.sum(posi)
	r=np.random.random()
	tot=0
	for i in xrange(len(note)):
		tot+=posi[i]
		if tot>=r: break
	return note[i]

def randomTrack(path='./converted'):
	'''
	Picks a random track from a random midi, then use it to initiallize the RNN.
	'''

	musiclist=os.listdir(path)
	idx=random.randint(0,len(musiclist)-1)
	while len(os.listdir(path+'\\'+musiclist[idx]))==0:
		idx=random.randint(0,len(musiclist)-1)
	filename=path+'\\'+musiclist[idx]
	tracklist=os.listdir(filename)
	filename+='\\'+tracklist[0]
	fp=open(filename,'r')
	lines=fp.readlines()
	seq=[]
	for i in xrange(1,len(lines)):
		seq.append(int(lines[i].split()[0]))
	return seq

'''
def randomTrack(path='./converted'):
	#Picks a random track from a random midi, then use it to initiallize the RNN.

	musiclist=os.listdir(path)
	idx=random.randint(0,len(musiclist)-1)
	while len(os.listdir(path+'\\'+musiclist[idx]))==0:
		idx=random.randint(0,len(musiclist)-1)
	filename=path+'\\'+musiclist[idx]
	tracklist=os.listdir(filename)
	filename+='\\'+tracklist[0]
	fp=open(filename,'r')
	lines=fp.readlines()
	seq=[]
	durs=[]
	resolution=float(lines[0])
	for i in xrange(1,len(lines)):
		seq.append(int(lines[i].split()[0]))
		durs.append(float(lines[i].split()[1]))
	return seq,durs,resolution

def getFirstHidden(note_net,beat_net,path='D:\\Artist\\converted'):

	seq,durs,resolution=randomTrack(path)
	while len(seq)==0 or len(durs)==0:
		seq,durs,resolution=randomTrack(path)
	length=min(int(len(seq)/3),int(len(durs)/3))
	durs=beatToVector(seq=durs,resolution=resolution)
	note_h=note_net.generateInitHidden()
	for note in seq:
		note_o,note_h=note_net.forward_pass(note_net.generateInputLayer(label=note),note_h)
	beat_h=beat_net.generateInitHidden()
	for measure in durs:
		beat_o,beat_h=beat_net.forward_pass(measure,beat_h)
	try:
		return note_h, seq[-1], beat_h, durs[-1]
	except IndexError:
		print(seq)
		print(durs)
'''



def getFirstNoteHidden(note_net):

	path='./converted'
	seq=randomTrack(path)
	while len(seq)==0:
		seq=randomTrack(path)
	length=int(len(seq)/3)
	note_h=note_net.generateInitHidden()
	for note in seq:
		note_o,note_h=note_net.forward_pass(note_net.generateInputLayer(label=note),note_h)
	return note_h, seq[-1]



def generateNoteSeq(net, hm1, init=60, length=100, choice_range=5):
	'''
	Generates a sequence of notes.
	param:
		net: a instance of Model, trained.
		hm1: the initial hidden layer of net, can be obtained by calling getFirstNoteHidden
	'''

	inp_dim=net.inp_dim
	hid_dim=net.hid_dim
	out_dim=net.out_dim

	res=[]
	last_note=init

	for i in xrange(length):
		inp_layer=net.generateInputLayer(last_note)
		out, hm1=net.forward_pass(inp_layer,hm1)
		note_list=[]
		posi_list=[]
		for choice in xrange(choice_range):
			note_list.append(np.argmax(out[0]))
			out[0][np.argmax(out[0])]=-1
			posi_list.append(out[0][np.argmax(out[0])])
		last_note=decide(note_list,posi_list)
		res.append(last_note)

	return res

'''
def generateBeatSeq(net, hm1, length=100, init=[[1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0]]):

	inp_dim=net.inp_dim
	hid_dim=net.hid_dim
	out_dim=net.out_dim

	res=[]
	inp_layer=init

	i=0
	while i<length:
		
		out, hm1=net.forward_pass(inp_layer,hm1)
		for j in xrange(len(out[0])):
			if out[0][j]>=0.5: out[0][j]=1
			else: out[0][j]=0
		i+=lenOfMeasure(out[0])
		for elem in out[0]:
			res.append(elem)
		inp_layer=out

	return res
'''


'''
The following two functions are called by run.py.
The function "withbeat" is currently under construction, so don't call it.
'''

def generateNoteSeqWithBeat(note_net, beat_net, start_note=60, length=100, choice_range=5):

	note_hm1, note_tail, beat_hm1, beat_tail= getFirstHidden(note_net,beat_net)
	return generateNoteSeq(note_net,note_hm1,init=note_tail),generateBeatSeq(beat_net,beat_hm1,init=beat_tail)

def generateNoteSeqWithoutBeat(note_net):
	'''
	Initiallize the RNN first, then generates a sequence of notes(a.k.a.melody).
	'''
	note_hm1, note_tail=getFirstNoteHidden(note_net)
	return generateNoteSeq(note_net,note_hm1,init=note_tail)