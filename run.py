from note import *
from load_music import *
import os,sys
import numpy as np
from note_seq import *
from songbird import *
from line import *

# NoteNet params:
note_iteration=100
note_inp_dim=89
note_hid_dim=50
note_out_dim=89
rate=0.005

# BeatNet params:
beat_iteration=100
beat_inp_dim=32
beat_hid_dim=20
beat_out_dim=32

music=loadMusic('converted')

def trainNoteNet(net,iteration):

	'''
	The complete process of training the note network.
	After every iteration, it prints out the value of "cand".
	This value is n, the actual next note of the training sample is the nth in the network's probability ranking.
	Example: the melody goes do mi so (fa) mi re do
	when the network has parsed do mi so, it predicts the next note(which should be fa) as follows:
	mi:40%, fa:30%, do:10%,...
	then cand=2
	The smaller it is, the better the network performs well on the training data.
	'''

	tot_len=0
	track_num=47
	tot_iter=0
	for piece in music:
		for track in piece.track:
			tot_iter+=len(track.pitch)-1
	for it in xrange(iteration):
		print('Begin iteration #%d'%it)
		tot_cand=0
		for piece in music:
			for track in piece.track:
				note_seq=track.pitch
				hm1=net.generateInitHidden()
				for i in xrange(len(note_seq)-1):
					inp_layer=net.generateInputLayer(note_seq[i])
					out,h= net.forward_pass(inp_layer,hm1)
					while np.argmax(out[0])!=note_seq[i+1]:
						out[0][np.argmax(out[0])]=-1
						tot_cand+=1
					d=net.generateDesired(note_seq[i+1])
					net.update(inp_layer,hm1,d)
					hm1=h
		print("Cand: "+str(float(tot_cand)/tot_iter))


def run(mode='file',iteration=-1):

	'''
	Run the complete process.

	param: mode= 'new' or 'file'.
	'file': load the trained network and generate midi from it.
	'new': train a new network which will be saved to disk later, and generate midi.

	iteration= number of iteration.
	'''

	if mode=='new':
		if iteration==-1:
			raise ValueError('Iteration times unspecified.')
		note_net=Model(mode=1,out_layer='softmax',idn=note_inp_dim,hdn=note_hid_dim,odn=note_out_dim,rate=rate)
		print iteration
		trainNoteNet(note_net,iteration=iteration)


	if mode=='file':
		note_net=Model(mode=2,out_layer='softmax',idn=note_inp_dim,hdn=note_hid_dim,odn=note_out_dim,rate=rate,
			filename='param_note.txt')
	notes=generateNoteSeqWithoutBeat(note_net)
	sing(notes)

	if mode=='new':
		note_net.saveToFile('param_note.txt')