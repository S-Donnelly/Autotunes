from music21 import converter, instrument, note, chord
from tqdm import tqdm
import glob
import time
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser(description='Process MIDI dataset.')
parser.add_argument('--dataset', type=str, nargs=1, required=True, help='Name of dataset in the dataset folder: Usage: bach_50')

parser.add_argument('--npz_file', type=str, nargs=1, required=True, help='.npz MIDI sequence save file: Usage: <dataset_name.npz>')

parser.add_argument('--network', type=str, nargs=1, required=True, choices=['AE','VAE'], help='The type of network the dataset will be used for: usage: <AE/VAE>')

parser.add_argument('--length', type=int, nargs=1, required=True, help='Length of processed sequences: Usage: <64>')

parser.add_argument('--trim', type=str, nargs=1, required=True,
					help='Argument to indicate if the script should process the entire songs or only the first sequence of <length> notes: Usage: <True/False>',
					choices=['True','False'])

args = parser.parse_args()

midi_data_dir = args.dataset[0]
npz_save_file = args.npz_file[0]
sequence_length = args.length[0]
cut_down = args.trim[0]

def process_and_save_notes():
	'''
	Process and save the midi dataset, extracts the note,chord and rest information of the midi
	files processes this information and save to a .npz file also outputs a
	'''

	notes = []
	sequence_cutoff = 0
	if cut_down == "True":
		sequence_cutoff = sequence_length
	else:
		sequence_cutoff = 100000

	for file in tqdm(glob.glob('datasets'+'\\'+midi_data_dir+"\\*.mid")):
		note_count = 0
		midi = converter.parse(file)
		notes_to_parse = None

		try: #if the midi file contains more than one instrument, process only the first instrument
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse()
		except:
			notes_to_parse = midi.flat.notes

		timesig = midi.getTimeSignatures()[0]
		ts = str(timesig.beatCount) + '/' + str(timesig.denominator)
		#if str(midi.analyze('key')) == 'C major' and ts == '4/4' :
		if check_length(notes_to_parse):
			for element in notes_to_parse: #.chordify():
				if note_count < sequence_cutoff:
					if isinstance(element, note.Note):
						duration = element.duration
						notes.append(str(element.pitch.ps) + '-' + str(duration.quarterLength))
						note_count = note_count + 1
					elif isinstance(element, chord.Chord):
						duration = element.duration
						notes.append('.'.join(str(int(n.pitch.ps)) for n in element) + '-' + str(duration.quarterLength))
						note_count = note_count + 1
					elif isinstance(element, note.Rest):
						notes.append(str(''))
						note_count = note_count + 1

	pitches = sorted(set(item for item in notes))
	print(len(notes))

	note_to_int = dict((note, number) for number, note in enumerate(pitches))

	network_inputs = []

	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		network_inputs.append([note_to_int[char] for char in sequence_in])

	print(len(network_inputs))

	max_val = max(max(network_inputs))
	print(max_val)
	min_val = min(min(network_inputs))
	feature_range = (0, max_val)
	predict_scalar = MinMaxScaler(feature_range=feature_range)

	network_inputs = np.asarray(network_inputs)
	normalised_inputs = network_inputs / max_val

	if args.network[0] == 'AE':
		division_point = int(normalised_inputs.shape[0] * 3/4)
		training, testing = normalised_inputs[0:division_point, :], normalised_inputs[division_point:-1:]
		np.savez('datasets' + '\\' + npz_save_file, x_train=training, x_test=testing, pitches=pitches, max_val=max_val)
	else:
		training = normalised_inputs
		np.savez('npz_datasets' + '\\' + npz_save_file, x_train=training, pitches=pitches, max_val=max_val)

def check_length(score):
	'''
	Check the length of the midi stream, only taking into all note, chord and rest
	objects
	'''
	num_ = 0
	score = score.flat.notes
	length = sequence_length
	for element in score:
		if isinstance(element, note.Note):
			num_ = num_ + 1
		elif isinstance(element, chord.Chord):
			num_ = num_ + 1
		elif isinstance(element, rest.Rest):
			num_ += 1
	if num_ < length:
		return False
	else:
		return True

if __name__ == '__main__':
	process_and_save_notes()
