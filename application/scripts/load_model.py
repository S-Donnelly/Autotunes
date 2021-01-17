from music21 import converter, instrument, chord, stream, duration, tempo, note, midi, duration
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Autoencoder training script')
parser.add_argument('--dataset', type=str, nargs=1, required=True, help='Name of the training npz file used during model training: Usage: <npz file name>')

parser.add_argument('--model', type=str, nargs=1, required=True, help='Name of model file from models folder: Usage: <model_file_name>')

parser.add_argument('--weights', type=str, nargs=1, required=True, help='Name of weights file from weights folder : Usage: <weights_file_name>')

parser.add_argument('--num_songs', type=int, nargs=1, required=True, help='Number of songs to be generated: Usage: <5>')

args = parser.parse_args()


t = datetime.now(tz=None)
timestamp = str(t.day) + '_' + str(t.month) + '_' + str(t.year) + '_' + str(t.hour) + str(t.minute) + str(t.second)

def load_model(model_, weights):
	'''
	Function to load the model architecture from the given .json and the weights from the .h5 file
	'''

	json_file = open('models\\' + model + ".json", "r")
	loaded_model_json = json_file.read()
	loaded_generator_model = model_from_json(loaded_model_json)
	loaded_generator_model.load_weights('models\\' + weights + ".h5")

	return loaded_generator_model

def save_song(song):
	'''
	Takes in the converted sequence and creates a midi stream, saves each note/chord and rest
	object to the stream then saves as a midi file in the VAE_generated_sequences fdir
	'''
	mt = midi.MidiTrack(0)
	dtime = midi.DeltaTime(mt)
	dtime.time = 0.5
	new_stream = stream.Stream()
	midi_save_dir = 'VAE_generated_sequences\\'

	for element in song:
		if ('.' in element) or element.isdigit():
			chord_component = element.split('-')
			duration_ = 1.0
			if '/' in chord_component[1]:
				duration_components = chord_component[1].split('/')
				duration_ = (int(duration_components[0])) / (int(duration_components[1]))
			else:
				duration_ = chord_component[1]

			notes_in_chord = chord_component[0].split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				notes.append(new_note)

			new_chord = chord.Chord(notes)
			new_chord.quarterLength = float(duration_)
			new_stream.append(new_chord)

		elif '.' not in element and (len(element) != 0):
			note_component = element.split('-')
			duration_ = None
			if '/' in note_component[1]:
				duration_components = note_component[1].split('/')
				duration_ = (int(duration_components[0])) / (int(duration_components[1]))
			else:
				duration_ = note_component[1]
			new_note = note.Note(int(note_component[0]))
			new_note.quarterLength = float(duration_)
			new_stream.append(new_note)
		elif element is "":
			new_stream.append(rest.Rest())

	count = len(os.listdir(midi_save_dir)) + 1
	midi_out = new_stream.write('midi', fp=midi_save_dir  +str(count) + '_' + dataset_name + '_' + str(100)
								+ '_' + str(timestamp) +".mid")

	print('-----------------------------')
	print('Sequence has been generated')
	print('-----------------------------')

def create_pattern(song_sequence, pitches):
	'''
	Pattern creation function that takes in the generated output of the network and
	converts it back into a list of values corrisponding to unique notes/chords/rests
	'''
	int_to_note = dict((number, note) for number, note in enumerate(pitches))
	prediction = []

	for note in song_sequence:
		if note < 0:
			note *= -1
		result = int_to_note[note[0]]
		prediction.append(result)
	return prediction

if __name__ == '__main__':
	dataset_name = args.dataset[0]
	latent_dim = 8
	network_data = np.load('npz_datasets' + '\\' + dataset_name + '.npz', allow_pickle=True)
	training_ = network_data['x_train']
	max_val = network_data['max_val']
	pitches = network_data['pitches']
	model = args.model[0]
	weights = args.weights[0]
	generator = load_model(model, weights)

	new_song = generator.predict(np.random.normal(loc=0.0, scale=1.0, size=(training_.shape[0], latent_dim)))
	for i in range(0, args.num_songs[0]):
		feature_range = (0, max_val)
		scalar = MinMaxScaler(feature_range=feature_range)
		a = new_song[i].reshape(-1,1)
		new_sequence = scalar.fit_transform(a).astype('int')
		new_sequence = create_pattern(new_sequence, pitches)
		save_song(new_sequence)
