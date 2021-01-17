from music21 import converter, instrument, note, chord
from tqdm import tqdm
import glob
import time
import numpy as np
import keras.utils as utils
import sys
from sklearn.preprocessing import MinMaxScaler
import argparse


parser = argparse.ArgumentParser(description='Process similar sequences of a song from a datasets.')
parser.add_argument('--dataset', type=str, nargs=1, required=True, help='Path to MIDI dataset folder: Usage: datasets\\<dataset_name>\\')

parser.add_argument('--song_dir', type=str, nargs=1, required=True, help='Path to MIDI file to try and match: Usage: datasets\\<dataset_name>\\<song_name.mid>')

parser.add_argument('--npz_file', type=str, nargs=1, required=True, help='.npz MIDI sequence save file: Usage: <name.npz>')

parser.add_argument('--length', type=int, nargs=1, required=True, help='Length of processed sequences: Usage: <64>')

args = parser.parse_args()


def process_and_save_notes():

	'''
	This function will process the supplied song by extracting the note,chord and rest information
	into sequences of the supplied length, then processes the supplied dataset in the same way,
	compares each sequence of the son to those from the midi dataset and saves any similar sequences to
	a .npz file
	'''

	all_notes = []
	midi_data_dir = args.dataset[0]
	npz_save_file = args.npz_file[0]
	midi_file_to_match = converter.parse(args.song_dir[0])
	sequence_length = args.length[0]

	note_sequences = []
	most_similar_sequences = []

	parts = instrument.partitionByInstrument(midi_file_to_match)
	if parts:
		notes_to_parse = parts.parts[0].recurse()
	else:
		notes_to_parse = midi.flat.notes
	sequences_to_match = create_sequences(notes_to_parse, sequence_length)

	for file in tqdm(glob.glob(midi_data_dir+"*.mid")):
		midi = converter.parse(file)
		notes_to_parse = None
		parts = instrument.partitionByInstrument(midi)
		if parts:
			notes_to_parse = parts.parts[0].recurse()
		else:
			notes_to_parse = midi.flat.notes
		temp_sequences = create_sequences(notes_to_parse, sequence_length)
		for n in temp_sequences:
			note_sequences.append(n)

	for seq in tqdm(sequences_to_match):
		for seq_2 in note_sequences:
			distance = check_distance(seq, seq_2)
			if (distance < sequence_length * 1/2):
				most_similar_sequences.append(seq_2)

	print(len(most_similar_sequences))
	notes = []
	for similar_seq in most_similar_sequences:
		for element in similar_seq:
			notes.append(element)

	print(len(notes))

	pitches = sorted(set(item for item in notes))

	note_to_int = dict((note, number) for number, note in enumerate(pitches))

	network_inputs = []

	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		network_inputs.append([note_to_int[char] for char in sequence_in])

	max_val = max(max(network_inputs))
	feature_range = (0, max_val)
	predict_scalar = MinMaxScaler(feature_range=feature_range)
	network_inputs = np.asarray(network_inputs)
	normalised_inputs = network_inputs / max_val

	np.savez('npz_datasets' + '\\' + npz_save_file, x_train=normalised_inputs, pitches=pitches, max_val=max_val)

def create_sequences(stream, seq_len):
	'''
	Takes a given midi stream and segments it into sequences and returns a list of those sequences
	'''
	length = get_midi_length(stream)
	sequences = []
	count = 0
	temp_sequence = []
	for element in stream:
		if count == seq_len:
			sequences.append(temp_sequence)
			temp_sequence = []
			count = 0
		else:
			count+=1
			if isinstance(element, note.Note):
				duration = element.duration
				temp_sequence.append(str(element.pitch.ps) + '-' + str(duration.quarterLength))
			elif isinstance(element, chord.Chord):
				duration = element.duration
				temp_sequence.append('.'.join(str(int(n.pitch.ps)) for n in element) + '-' + str(duration.quarterLength))
			elif isinstance(element, note.Rest):
				temp_sequence.append('')
	return sequences

def get_midi_length(score):
	'''
	Checks the number of note, chord and rest objects in a midi stream and returns the result
	'''
	length = 0
	for element in score:
		if isinstance(element, note.Note):
			length+=1
		elif isinstance(element, chord.Chord):
			length+=1
		elif isinstance(element, note.Rest):
			length+=1
	return length

def check_distance(sequence_1, sequence_2):
	# Using the Levenshtein distance algorithim this method aims to find similar sequences within the datase
	'''
	This function uses the Levenshtein distance algorithim to check the minimum number of changes that must be made
	to sequence_2 to change it to sequence_1 
	'''
	size_x = len(sequence_1) + 1
	size_y = len(sequence_2) + 1
	matrix = np.zeros((size_x, size_y))

	for x in range(1, size_x):
		matrix[x, 0] = x
	for y in range(1, size_y):
		matrix[0, y] = y

	for x in range(1, size_x):
		for y in range(1, size_y):
			if sequence_1[x-1] == sequence_2[y-1]:
				matrix[x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
			else:
				matrix[x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1] + 1, matrix[x, y-1] + 1)

	return (matrix[size_x - 1, size_y - 1])

if __name__ == '__main__':
	process_and_save_notes()
