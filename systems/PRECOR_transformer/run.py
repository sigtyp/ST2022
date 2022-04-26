# If reproducibility is required, run the following command:
# CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python run.py

import os
import random
import csv
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sigtypst2022 import sigtypst2022_path, compare_words
from tensorflow import keras
from tensorflow.keras import layers

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

def open_file(path):
	with open(path) as f:
		reader = csv.DictReader(f, delimiter='\t')
		dictionaries = list(reader)
		return dictionaries

def create_training_x_y(training, target_column):
	lang_name = []
	training_texts = []
	for n, v in enumerate(training):
		g = list(v.items())
		t = g[target_column][1]
		if t == "":
			pass
		else:
			t = "$ " + t + " #"
			for n, (k, v) in enumerate(g):
				if n == target_column or n == 0:
					pass
				elif v == "":
					pass
				else:
					training_texts.append((v, t))
					lang_name.append([k]) # [] for scikit
	return training_texts, lang_name

def create_test_x(test, number_column_x):
	test_x = []
	lang_name = []
	cogid = []
	for n, j in enumerate(test):
			g = list(j.items())
			t = g[number_column_x][1]
			if t == "?":
				for n, (k, v) in enumerate(g):
					if n == 0:
						cogid.append(v)
					elif v == "?":
						pass
					else:
						test_x.append(v)
						lang_name.append([k])						
	return test_x, lang_name, cogid

def create_test_y(test, number_column_y, n_max_colmn):
	test_y = []
	for j in test:
		g = list(j.items())
		t = g[number_column_y][1]
		if t != "":
				for n, (k, v) in enumerate(g):
					if n == 0 or v =="":
						pass
					else:
						for x in range(0, n_max_colmn - 1):
							test_y.append(v)
	return test_y

def decode(data, dictionary):
	all_decoded = []
	for x in data:
		decoded_sentence = []
		for y in x:
			nb = dictionary[np.argmax(y)]
			if nb == "#":
				break
			else:
				decoded_sentence.append(nb)
		all_decoded.append(decoded_sentence)
	return all_decoded

def transform_inverse(predicted_test, n_max_clm):
	tensor_final = np.zeros((int(predicted_test.shape[0] / (n_max_clm -1)),  
							max_training_x_seq_length, predicted_test.shape[2]))
	for n, v in enumerate(range(0,predicted_test.shape[0], n_max_clm -1)): 
		 new = np.mean(predicted_test[v : v + n_max_clm -1], axis=0)
		 tensor_final[n] = new
	return tensor_final

def print_tsv(nmber_col_max, nmber_col, test_x_cogid, predicted_test):
	all_lines = []
	for c, n in zip(test_x_cogid, predicted_test):
		nj = n.replace(" #", "")
		nj = nj.replace("$ ", "")
		nj = nj.rstrip()
		#nj = nj.replace("[UNK]", "@")
		tab_before = nmber_col - 1
		tab_before2 = []
		for tab in range(tab_before):
			tab_before2.append("")
		tab_after = nmber_col_max - nmber_col
		tab_after2 = []
		for tab in range(tab_after):
			tab_after2.append("")
		joinall = "\t".join((c, *tab_before2, nj, *tab_after2))
		all_lines.append(joinall)
	return all_lines

def format_dataset(x, y, languages):
    x = train_x_vectorization(x)
    y = train_y_vectorization(y)
    return ({"encoder_inputs": x, 
			 "decoder_inputs": y[:, :-1], 
			 "languages": languages}, y[:, 1:])

def make_dataset(pairs, languages, shuffle=True):
    x, y = zip(*pairs)
    x = list(x)
    y = list(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y, languages))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    if shuffle == True:
    	return dataset.shuffle(2048).prefetch(16)
    else:
    	return dataset.prefetch(16)

# This is the Keras transformer architecture presented at
# https://github.com/keras-team/keras-io/blob/master/examples/nlp/
# neural_machine_translation_with_transformer.py
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
          [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], 
            	dtype=tf.int32)], axis=0,
                         )
        return tf.tile(mask, mult)

def decode_sequence(predictions, inverse_vocabulary):
	all_sentences = []
	for a in range(predictions.shape[0]):
	    decoded_sentence = "$"
	    for i in range(predictions.shape[1]):
	        sampled_token_index = np.argmax(predictions[a, i, :])
	        sampled_token = inverse_vocabulary[sampled_token_index]
	        decoded_sentence += " " + sampled_token
	        if sampled_token == "#":
	            break
	    all_sentences.append(decoded_sentence)
	return all_sentences

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="PRECOR-Transformer for Prediction of Cognate Reflexes")
	parser.add_argument("--surprise", dest="surprise", action="store_true", help="Train using surprise data")
	parser.add_argument("--embed_dimensions", dest="embed_dim", default=150, type=int)
	parser.add_argument("--latent_dimensions", dest="latent_dim", default=800, type=int)
	parser.add_argument("--heads", dest="num_heads", default=5, type=int)
	parser.add_argument("--batch_size", dest="batch_size", default=400, type=int)

	args = parser.parse_args()

	embed_dim = args.embed_dim
	latent_dim = args.latent_dim
	num_heads = args.num_heads
	batch_size = args.batch_size

	if args.surprise:
		modality = "surprise"
		path_reader = "data-surprise"
	else:
		modality = "training"
		path_reader= "data"

	percentages = ["0.10", "0.20", "0.30", "0.40", "0.50"]

	for per in percentages:
		print("\n***Training for folder: '" + modality +"', percentage: " + per + "***\n")
		path_train = sigtypst2022_path(path_reader).glob("*/training-{}.tsv".format(per))
		path_test_x = sigtypst2022_path(path_reader).glob("*/test-{}.tsv".format(per))
		path_test_y = sigtypst2022_path(path_reader).glob("*/solutions-{}.tsv".format(per))
		baseline = sigtypst2022_path(path_reader).glob("*/result-{}.tsv".format(per))

		list_files=list(zip(path_train, path_test_x, path_test_y, baseline))

		for a1,b1,c1,d1 in list_files:
			
			training, testX, testY = open_file(a1), open_file(b1), open_file(c1)
			max_training_x_seq_length=max([len(y)for x in training for y in x.values()])

			name_dir = os.path.split(os.path.split(a1)[0])[1]
			
			MAX_NMB_CLMN = len(training[0]) - 1

			mfile = []

			for NMB_CLMN in range(1, MAX_NMB_CLMN + 1):
				print("\n\n***Training for languages in " + name_dir + \
							      ", target column: " + str(NMB_CLMN) + "***\n")
				training_texts, training_x_lang_names = create_training_x_y(training, NMB_CLMN)
				train_x_texts = [pair[0] for pair in training_texts]
				train_y_texts = [pair[1] for pair in training_texts]

				max_training_x_seq_length = max([len(y) for x in training_texts for y in x])
				phonemes_x = [y.split(" ") for x in training_texts for y in x]
				vocabulary_x = list(set([y for x in phonemes_x for y in x]))
				char_size = len(vocabulary_x)

				train_x_vectorization = layers.TextVectorization(
					output_mode="int", 
					           output_sequence_length=max_training_x_seq_length,
					standardize=None, split=lambda x: tf.strings.split(x, sep=" "))	
				train_y_vectorization = layers.TextVectorization(
					output_mode="int", 
					       output_sequence_length=max_training_x_seq_length + 1,
					standardize=None, split=lambda x: tf.strings.split(x, sep=" "))			

				train_x_vectorization.adapt(train_x_texts)
				train_y_vectorization.adapt(train_y_texts)
		
				enc = OneHotEncoder(handle_unknown='ignore')
				enc.fit(training_x_lang_names)
				tensor_training_x_lang = enc.transform(training_x_lang_names).toarray()
				tensor_training_x_lang = tensor_training_x_lang.reshape(tensor_training_x_lang.shape[0], 1,tensor_training_x_lang.shape[1])
				tensor_training_x_lang = np.repeat(tensor_training_x_lang, max_training_x_seq_length, axis=1)
				couple = list(zip(training_texts, tensor_training_x_lang))
				random.shuffle(couple)
				training_texts, tensor_training_x_lan = zip(*couple)

				train_ds = make_dataset(training_texts, tensor_training_x_lang, shuffle=True)

				test_x_texts, test_x_lang_names, test_cogid = create_test_x(testX, NMB_CLMN)
				test_y_texts = create_test_y(testY, NMB_CLMN, MAX_NMB_CLMN)
				test_y_texts = ["$ " + x + " #" for x in test_y_texts]
				test_texts = [(x, y) for x, y in zip(test_x_texts, test_y_texts)]
				tensor_test_x_lang = enc.transform(test_x_lang_names).toarray()
				tensor_test_x_lang = tensor_test_x_lang.reshape(tensor_test_x_lang.shape[0], 1,tensor_test_x_lang.shape[1])
				tensor_test_x_lang = np.repeat(tensor_test_x_lang, max_training_x_seq_length , axis=1)

				test_ds = make_dataset(test_texts, tensor_test_x_lang, shuffle=False)

				# Model

				encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
				x = PositionalEmbedding(max_training_x_seq_length, char_size, embed_dim)(encoder_inputs)
				encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
				x = layers.Dropout(0.6)(x)
				encoder = keras.Model(encoder_inputs, encoder_outputs)

				decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
				encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
				x = PositionalEmbedding(max_training_x_seq_length, char_size, embed_dim)(decoder_inputs)
				x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
				x = layers.Dropout(0.6)(x)
				
				languages_inputs = keras.Input(shape=(None, tensor_training_x_lang.shape[2]), name="languages")
				x = tf.keras.layers.concatenate([x, languages_inputs])

				decoder_outputs = layers.Dense(char_size, activation="softmax")(x)
				
				decoder = keras.Model([decoder_inputs, encoded_seq_inputs, languages_inputs], decoder_outputs)

				decoder_outputs = decoder([decoder_inputs, encoder_outputs, languages_inputs])
				
				transformer = keras.Model([encoder_inputs, decoder_inputs, languages_inputs], decoder_outputs, name="transformer")

				epochs = 200  # This should be at least 30 for convergence
				callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5)

				transformer.summary()
				print("\n\n*** Target column number: " + \
					  "{n1} out of {n2} ***\n\n".format(n1=NMB_CLMN, n2=MAX_NMB_CLMN))
				transformer.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
				   loss="sparse_categorical_crossentropy", metrics=["accuracy"]
				)
				transformer.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=callback,)
				transformer.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
				   loss="sparse_categorical_crossentropy", metrics=["accuracy"]
				)
				transformer.fit(train_ds, epochs=epochs, validation_data=test_ds, callbacks=callback,)

				# decode
				train_y_vocab = train_y_vectorization.get_vocabulary()
				train_y_vocab_index_lookup = dict(zip(range(len(train_y_vocab)), train_y_vocab))
				max_decoded_sentence_length = max_training_x_seq_length

				predicted_test2 = transformer.predict(test_ds)
				predicted_test3 = transform_inverse(predicted_test2, MAX_NMB_CLMN)
				aaa = decode_sequence(predicted_test3, train_y_vocab_index_lookup)
				pr = print_tsv(MAX_NMB_CLMN, NMB_CLMN, test_cogid, aaa)	
				
				mfile.append(pr)

			header = list(training[0].keys())
			header = "\t".join(header) 
			body = "\n".join([y for x in mfile for y in x])
			hb = "\n".join([header, body])
			print("\n")
			print(hb)
			print("\n")
			os.makedirs(os.path.join(sigtypst2022_path(), "systems", "PRECOR_transformer", modality, name_dir), exist_ok=True)
			mypath = sigtypst2022_path("systems", "PRECOR_transformer", modality, name_dir, "result-{}.tsv".format(per))
			file = open(mypath, "w")
			print(hb, file=file)
			file.close()
			print("***Parser results***\n")
			compare_words(mypath, c1)
			print("\n***Baseline results***\n")
			compare_words(d1, c1)
