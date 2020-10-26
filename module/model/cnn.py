import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
#!wget http://nlp.stanford.edu/data/glove.6B.zip

class CNN(object):
	def __init__(self, classes, config):
		self.models = {}
		self.config = config
		self.classes = classes
		for cls in self.classes:
			model = Sequential()
			model_train = self.defined_cnn(self.config, model)
			print(type(model_train))
			self.models[cls] = model_train

	def fit(self, train_x, train_y):
		# enumerate :https://www.geeksforgeeks.org/enumerate-in-python/
		for idx, cls in enumerate(self.classes):
			class_labels = train_y[:,idx]
			self.models[cls].fit(train_x, class_labels, batch_size=self.config['batch_size'],epochs=self.config['epochs'])

	def predict(self, test_x):
		predictions = np.zeros((test_x.shape[0], len(self.classes)))
		# print(".............")
		# print(predictions.shape)
		# print(".............")
		for idx, cls in enumerate(self.classes):
			predictions[:, idx] = self.models[cls].predict(test_x,verbose=self.config['verbose'],batch_size=self.config['pred_batch_size'])
		return predictions

	def predict_prob(self, test_x):
		probs = np.zeros((test_x.shape[0], len(self.classes)))
		for idx, cls in enumerate(self.classes):
            # only want the probability of getting the output either as 0 or 1
            # Ref: https://discuss.analyticsvidhya.com/t/what-is-the-difference-between-predict-and-predict-proba/67376/3
			probs[:, idx] = self.models[cls].predict_proba(test_x)[:,1]
		return probs


	def defined_cnn(self, config, model):
		model = Sequential()
		model.add(Embedding(20000,
    		self.config['embedding_dim'],
    		embeddings_initializer = tf.keras.initializers.constant(
    			self._embedding_layer(text.Tokenizer(20000))),
    		trainable = False))
		model.add(Dropout(self.config['dropout_rate']))
		model.add(Conv1D(self.config['filters'],
    		self.config['kernel_size'],
    		padding = 'valid'))
		model.add(MaxPooling1D())
		model.add(Conv1D(self.config['filters'],
    		5,
    		padding = 'valid',
    		activation = 'relu'))
		model.add(GlobalMaxPooling1D())
		model.add(Dense(self.config['hidden_dims'],
    		activation = 'relu'))
		model.add(Dropout(self.config['dropout_rate']))
		model.add(Dense(1,activation='sigmoid'))
		model.summary()
		model.compile(loss=self.config['loss'],
    		optimizer=self.config['optimizer'],
    		metrics=self.config['metrics'])

		return model

	def _embedding_layer(self, x_tokenizer):

		embeddings_index = dict()
		f = open(self.config['embedding_file_input'])
		for line in f:
			values = line.split()
			word = values[0]
			coef = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coef
		print(f'Found {len(embeddings_index)} word vectors.')

		embedding_matrix=np.zeros((20000,self.config['embedding_dim']))
		for word, idx in x_tokenizer.word_index.items():
			if idx > self.config['max_features']-1:
				break
			else:
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					embedding_matrix[idx] = embedding_vector

		
		return embedding_matrix		
