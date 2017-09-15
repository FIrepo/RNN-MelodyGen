import midi
import math


def sing(seq):
	'''
	It turns a sequence of pitches into a midi.
	You can alternate the resolution to change its speed.
	'''

	pattern=midi.Pattern()
	track = midi.Track()

	pattern.append(track)
	#resolution
	pattern.resolution=200
	for note in seq:
		on = midi.NoteOnEvent(tick=0, velocity=50, pitch=note)
		track.append(on)
		off = midi.NoteOffEvent(tick=100, pitch=note)
		track.append(off)
	track.append(midi.EndOfTrackEvent(tick=1))
	midi.write_midifile("song.mid", pattern)

'''
def singWithBeat(note,beat):

	print(note)
	print(beat)

	pattern=midi.Pattern()
	track = midi.Track()

	pattern.append(track)
	i=0
	noteidx=0
	while i<len(beat) and noteidx<len(note):
		on = midi.NoteOnEvent(tick=0, velocity=50, pitch=note[noteidx])
		track.append(on)
		dur=1
		while i<len(beat) and beat[i]==1:
			i+=1
			dur+=1
		i+=1
		off = midi.NoteOffEvent(tick=400*dur/32, pitch=note[noteidx])
		noteidx+=1
		track.append(off)
	track.append(midi.EndOfTrackEvent(tick=1))
	midi.write_midifile("song.mid", pattern)
'''