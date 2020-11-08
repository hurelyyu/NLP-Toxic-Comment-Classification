import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization

#!wget http://nlp.stanford.edu/data/glove.6B.zip

class CNN(object):
	def __init__(self, classes, config, pretrained_embedding):

		self.config = config
		self.classes = classes
		self.num_class = len(classes)
		self.model = self.defined_cnn(pretrained_embedding)

	def fit(self, train_x, train_y):
		# enumerate :https://www.geeksforgeeks.org/enumerate-in-python/
		# for idx, cls in enumerate(self.classes):
		# 	class_labels = train_y[:,idx]
		# 	self.models[cls].fit(train_x, class_labels, batch_size=self.config['batch_size'],epochs=self.config['epochs'],verbose = True)
		fully_model = self.model.fit(train_x, train_y,
								 epochs=self.config['epochs'],
								 verbose=True,
								 # validation_data=(validate_x, validate_y),
								 batch_size=self.config['batch_size'])
		return fully_model


	def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
		history = self.model.fit(train_x, train_y,
								 epochs=self.config['epochs'],
								 verbose=True,
								 # validation_data=(validate_x, validate_y),
								 batch_size=self.config['batch_size'])
		predictions = self.predict(validate_x)
		return predictions, history


	def predict(self, test_x):
		probs = self.model.predict(test_x)
		return probs >= 0.5
		# predictions = np.zeros((test_x.shape[0], len(self.classes)))
		# # print(".............")
		# # print(predictions.shape)
		# # print(".............")
		# for idx, cls in enumerate(self.classes):
		# 	predictions[:, idx] = self.models[cls].predict(test_x,verbose=self.config['verbose'],
		# 												   batch_size=self.config['pred_batch_size']).flatten()
		# return predictions

	def predict_prob(self, test_x):
		# probs = np.zeros((test_x.shape[0], len(self.classes)))
		# for idx, cls in enumerate(self.classes):
         #    # only want the probability of getting the output either as 0 or 1
         #    # Ref: https://discuss.analyticsvidhya.com/t/what-is-the-difference-between-predict-and-predict-proba/67376/3
		# 	probs[:, idx] = self.models[cls].predict_proba(test_x)[:,1]
		# return probs
		return self.model.predict(test_x)

	def defined_cnn(self, pretrained_embedding):
		model = Sequential()
		if pretrained_embedding is not None:
			model.add(Embedding(self.config['vocab_size'],
								self.config['embedding_dim'],
								# embeddings_initializer="uniform",
								weights=[pretrained_embedding],
								trainable=False, input_length=self.config['maxlen']))
		else:
			model.add(Embedding(self.config['vocab_size'],
					self.config['embedding_dim'],
					embeddings_initializer="uniform",
				# embeddings_initializer = tf.keras.initializers.constant(
					# self._embedding_layer(text.Tokenizer(20000))),
					trainable = True, input_length=self.config['maxlen']))


		model.add(Conv1D(128,
    		7,
    		padding = 'same',activation='relu'))
		model.add(BatchNormalization())
		model.add(MaxPooling1D())
		#model.add(BatchNormalization())
		model.add(Conv1D(256,
    		5,
    		padding = 'same',
    		activation = 'relu'))
		model.add(GlobalMaxPooling1D())
		#model.add(BatchNormalization())
		model.add(Dense(self.config['hidden_dims'],
    		activation = 'relu'))
		model.add(Dropout(self.config['dropout_rate']))
		model.add(BatchNormalization())
		model.add(Dense(self.num_class, activation=None))
		model.add(Dense(self.num_class,activation='sigmoid'))
		model.compile(loss=self.config['loss'],
    		optimizer=self.config['optimizer'],
    		metrics=self.config['metrics'])
		model.summary()

		return model

	# def _embedding_layer(self, x_tokenizer):
    #
	# 	embeddings_index = dict()
	# 	f = open(self.config['embedding_file_input'])
	# 	for line in f:
	# 		values = line.split()
	# 		word = values[0]
	# 		coef = np.asarray(values[1:], dtype='float32')
	# 		embeddings_index[word] = coef
	# 	print(f'Found {len(embeddings_index)} word vectors.')
    #
	# 	embedding_matrix=np.zeros((20000,self.config['embedding_dim']))
	# 	for word, idx in x_tokenizer.word_index.items():
	# 		if idx > self.config['max_features']-1:
	# 			break
	# 		else:
	# 			embedding_vector = embeddings_index.get(word)
	# 			if embedding_vector is not None:
	# 				embedding_matrix[idx] = embedding_vector
    #
    #
	# 	return embedding_matrix
