import os
from music21 import converter, instrument, chord, stream, duration, tempo, note, midi, duration
import tensorflow as tf
from keras import backend as K
import numpy as np
from tqdm import tqdm
import keras as ks
from keras.losses import mse, binary_crossentropy
import glob
from sklearn.preprocessing import MinMaxScaler
import keras as keras
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

'''
Command line argument parser
'''
parser = argparse.ArgumentParser(description='Variational Autoencoder Training Script')
parser.add_argument('--dataset', type=str, nargs=1, required=True, help='Name of the training npz file: Usage: <bach_s50_sq64>')
parser.add_argument('--batch_size', type=int, nargs=1, required=True, help='Number of samples to be processed per iteration: Note: Recommended to be some multiple of sequence size but less than total number of samples: Usage: <12>')
parser.add_argument('--epochs', type=int, nargs=1, required=True, help='Number of training epochs for the network: Usage: <50>')
args = parser.parse_args()

'''
Defining and loading the training dataset
'''
dataset_name = args.dataset[0]
network_data = np.load('npz_datasets' + '\\' + dataset_name + '.npz', allow_pickle=True) #Load .npz file
training_ = network_data["x_train"]	#Extract training data
pitches = network_data["pitches"] #Extract list of pitches
max_val = network_data["max_val"] #Extract maximum value of the pitches
training_ = training_.astype('float32')

t = datetime.now(tz=None)
timestamp = str(t.day) + '_' + str(t.month) + '_' + str(t.year) + '_' + str(t.hour) + str(t.minute) + str(t.second)

'''
Defining the networks parameters, number of training epochs and the save directories for the generated sequences,
the trained models and the loss graph.
'''
params = {'input_dim' : training_.shape[1],
		  'hidden_dim' : 42,
		  'hidden_dim_2' : 28,
		  'hidden_dim_3' : 18,
		  'hidden_dim_4' : 12,
		  'latent_dim' : 8,
		  'epochs' : args.epochs[0],
		  'batch_size' : args.batch_size[0],
		  'save_dir' : 'VAE_generated_sequences\\',
		  'model_save_dir' : 'models',
		  'graph_save_dir' : 'graphs' }

def sampling(params):
	'''
	Sampling function that take in a mean and standard deviation and returns a sample
	from a normal Guassian curve
	'''
	mean, logVar = params
	batchSize = tf.shape(mean)[0]
	latentDim = tf.shape(mean)[1]
	epsilon = tf.random.normal(shape=(batchSize, latentDim), mean=0.0, stddev=0.001)
	return mean + tf.exp(logVar / 2.0) * epsilon

def create_pattern(song_sequence, pitches):
	'''
	Pattern creation function that takes in the generated output of the network and
	converts it back into a list of values corrisponding to unique notes/chords/rests
	'''
	int_to_note = dict((number, note) for number, note in enumerate(pitches))
	prediction = []

	for note in song_sequence:
		result = int_to_note[note[0]]
		prediction.append(result)
	return prediction

def save_model(model):
	'''
	This function saves the decoder model of the network, creates a .json and .h5 file
	in the models directories
	'''
	model_json = model.to_json()
	with open(params['model_save_dir'] + '\\' + 'generator_' + dataset_name + '_vae_model_' + str(timestamp) + '.json', "w") as json_file:
		json_file.write(model_json)
	model.save(params['model_save_dir'] + '\\' + 'generator_' + dataset_name + '_vae_weights_' + str(timestamp) +'.h5')
	print('-------------------------------------------------')
	print('Model has been saved')
	print('-------------------------------------------------')

def plot_graph(history, item):
	'''
	This function plots the loss graph over the training epochs
	'''
	plt.plot(history.history[item])
	plt.title('Model Loss')
	plt.ylabel('Loss Value')
	plt.xlabel('Epochs')
	plt.savefig(params['graph_save_dir'] + '\\' + dataset_name + '_vae_' + str(timestamp))
	plt.show()

def save_song(song):
	'''
	Takes in the converted sequence and creates a midi stream, saves each note/chord and rest
	object to the stream then saves as a midi file in the VAE_generated_sequences fdir
	'''

	mt = midi.MidiTrack(0)
	dtime = midi.DeltaTime(mt)
	dtime.time = 0.5
	new_stream = stream.Stream()

	for element in song:
		if ('.' in element) or element.isdigit():
			chord_component = element.split('-')
			if '/' in chord_component[1]:
				duration_components = chord_component[1].split('/')
				duration_ = (float(duration_components[0])) / (float(duration_components[1]))
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
				duration_ = (float(duration_components[0])) / (float(duration_components[1]))
			else:
				duration_ = note_component[1]
			new_note = note.Note(int(note_component[0]))
			new_note.quarterLength = float(duration_)
			new_stream.append(new_note)
		elif element is "":
			new_stream.append(note.Rest())

	count = len(os.listdir(params['save_dir'])) + 1
	midi_out = new_stream.write('midi', fp=params['save_dir']  +str(count) + '_' + dataset_name + '_' + str(params['input_dim'])
								+ '_' + str(timestamp) +".mid")
	print("--------------------------------------------------")
	print("Generated Sequence Saved")
	print("--------------------------------------------------")

def build_network():
	'''
	Builds and trains the neural network model, returns the trained vae model, the
	decoder model and the training history
	'''

	input_s = keras.layers.Input(shape=((params['input_dim'],)))

	x = keras.layers.Dense(params['hidden_dim'], activation='relu')(input_s)

	x = keras.layers.Dense(params['hidden_dim_2'], activation='relu')(x)

	x = keras.layers.Dense(params['hidden_dim_3'], activation='relu')(x)

	x = keras.layers.Dense(params['hidden_dim_4'], activation='relu')(x)

	mean = keras.layers.Dense(params['latent_dim'], activation='relu')(x)

	log_variance = keras.layers.Dense(params['latent_dim'], activation='relu')(x)

	encoded = keras.layers.Lambda(sampling, output_shape=(params['latent_dim'],))([mean, log_variance])

	latent = keras.layers.Input(shape=(params['latent_dim'],))

	x = keras.layers.Dense(params['hidden_dim_4'], activation='relu')(latent)

	x = keras.layers.Dense(params['hidden_dim_3'], activation='relu')(x)

	x = keras.layers.Dense(params['hidden_dim_2'], activation='relu')(x)

	x = keras.layers.Dense(params['hidden_dim'], activation='relu')(x)

	decoded = keras.layers.Dense(params['input_dim'], activation='sigmoid')(x)

	encoder = keras.Model(input_s, [mean, log_variance, encoded])

	decoder = keras.Model(latent, decoded)

	decoded = decoder(encoder(input_s)[0])

	vae = keras.Model(input_s, decoder(encoded))

	re_loss = tf.keras.losses.mean_squared_error(K.flatten(vae.inputs[0]), K.flatten(vae.outputs[0]))

	klLoss = 1.5 * (1 + log_variance - K.square(mean) - K.exp(log_variance))

	klLoss = K.sum(klLoss, axis=-1)

	klLoss *= -0.5

	vae_loss = K.mean(re_loss + klLoss)

	vae.add_loss(vae_loss)

	vae.summary()

	optimiser = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False)

	#vae.compile(optimizer='rmsprop')
	vae.compile(optimizer=optimiser)

	history = vae.fit(training_, epochs=params['epochs'], batch_size=params['batch_size'])

	return vae, decoder, history

if __name__ == '__main__':
	model, decoder, history = build_network()
	plot_graph(history, 'loss')
	save_model(decoder)

	'''
	generates 3 new sequences from the decoder model
	'''
	new_song = decoder.predict(np.random.normal(loc=0.0, scale=1.0, size=(training_.shape[0], params['latent_dim'])))
	for i in range(0, 3):
		feature_range = (0, max_val)
		scalar = MinMaxScaler(feature_range=feature_range)
		a = new_song[i].reshape(-1,1)
		new_sequence = scalar.fit_transform(a).astype('int')
		new_sequence = create_pattern(new_sequence, pitches)
		save_song(new_sequence)
