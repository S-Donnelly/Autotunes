import os
from music21 import converter, instrument, chord, stream, duration, tempo, note, midi
import tensorflow as tf
from keras import backend
import keras as ks
import numpy as np
from tqdm import tqdm
import keras as ks
from keras.losses import mse, binary_crossentropy
import glob
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Autoencoder training script')
parser.add_argument('--dataset', type=str, nargs=1, required=True, help='Name of the training npz file: Usage: <bach_s50_sq64>', default='bach_s50_sq64')
parser.add_argument('--latent_dim', type=int, nargs=1, required=True, help='Size of the network latent dimension: Note: Recommended to be 1/4 of training sequence size: Usage: <12>',
 					default=12)
parser.add_argument('--epochs', type=int, nargs=1, required=True, help='Number of training epochs for the network: Usage: <50>', default=50)

args = parser.parse_args()

dataset_name = args.dataset[0]
network_data = np.load('npz_datasets' + '\\' + dataset_name + '.npz', allow_pickle=True)
training_ = network_data["x_train"]
testing_ = network_data["x_test"]
pitches = network_data["pitches"]
max_val = network_data["max_val"]
training_ = training_.astype('float32')

t = datetime.now(tz=None)
timestamp = str(t.day) + '_' + str(t.month) + '_' + str(t.year) + '_' + str(t.hour) + str(t.minute) + str(t.second)

params = {'input_dim' : training_.shape[1],
		  'latent_dim' : args.latent_dim[0],
		  'epochs' : args.epochs[0],
		  'batch_size' : 64,
		  'save_dir' : 'AE_generated_sequences\\',
		  'model_save_dir' : 'models',
		  'graph_save_dir' : 'graphs' }

def train_network():
	model = build_network(training_)
	trained_model = train_model(model, training_)
	generate(trained_model, testing_, pitches, max_val)

def build_network(network_input):
	encoderInput = tf.keras.layers.Input(shape=(params['input_dim'],))
	latent = tf.keras.layers.Input(shape=(params['latent_dim']))
	encoded = tf.keras.layers.Dense(params['latent_dim'], activation = 'relu')(encoderInput)
	decoded = tf.keras.layers.Dense(params['input_dim'], activation = 'sigmoid')(latent)
	encoder = tf.keras.Model(encoderInput, encoded)
	decoder = tf.keras.Model(latent, decoded)
	autoencoder = tf.keras.Model(encoderInput, decoder(encoded))
	autoencoder.summary()
	return autoencoder

def train_model(model, network_input):
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	model.fit(network_input, network_input, epochs=params['epochs'], batch_size=params['batch_size'])
	print("--------------------------------------------------------")
	print("Finished Training")
	print("--------------------------------------------------------")

	return model

def generate(model, testing_, pitches, max_val):
	feature_range = (0, int(max_val))
	scalar = MinMaxScaler(feature_range=feature_range)
	decoded_song = model.predict(testing_)
	songs = (scalar.fit_transform(decoded_song)).astype('int')
	original_songs = (scalar.fit_transform(testing_)).astype('int')
	new_song = create_pattern(songs[0], pitches)
	original_song = create_pattern(original_songs[0], pitches)
	save_song(new_song , "generated")

def create_pattern(song_sequence, pitches):
	int_to_note = dict((number, note) for number, note in enumerate(pitches))
	prediction = []
	for note in song_sequence:
		result = int_to_note[note]
		prediction.append(result)

	return prediction

def save_song(song , folder):
	mt = midi.MidiTrack(0)
	dtime = midi.DeltaTime(mt)
	dtime.time = 0.5
	new_stream = stream.Stream()
	new_stream.duration.quarterLength
	save_dir = params['save_dir'] + folder + '\\'

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

	name = folder + 'file_'
	count = len(os.listdir(save_dir)) + 1
	midi_out = new_stream.write('midi', fp=save_dir + name + str(count) + ".mid")

if __name__ == '__main__':
	train_network()
