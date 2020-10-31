import io
import re, os, sys, string, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer, word_tokenize, StanfordSegmenter, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D



class Preprocessor_cnn(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = self.config['classes']
        self._load_data()

    def _parse(self, data_frame, is_test=False):
        X = data_frame[self.config['input_text_column']].apply(Preprocessor_cnn.clean_txt).values
        Y = None
        if not is_test:
            Y = data_frame.drop([self.config['input_id_column'], self.config['input_text_column']], 1).values
        else:
            Y = data_frame.id.values

        return X, Y

    def _tokenize(self, train):
        x_tokenizer = text.Tokenizer(self.config['max_features'])
        x_tokenizer.fit_on_texts(list(train))
        x_tokenized = x_tokenizer.texts_to_sequences(train)
        x = sequence.pad_sequences(x_tokenized,maxlen=self.config['max_text_length'])

        return x

    def _test_tokenize(self, test):
        x_tokenizer = text.Tokenizer(self.config['max_features'])
        x_tokenizer.fit_on_texts(list(test))
        x_test_tokenized=x_tokenizer.texts_to_sequences(test)
        test = sequence.pad_sequences(x_test_tokenized,maxlen=self.config['max_text_length'])

        return test

    def _load_data(self):
        train_df = pd.read_csv(self.config['input_trainset'])
        #train_df['none'] = 1 - train_df[self.classes].max(axis=1)
        self.train_df_x, self.train_df_y = self._parse(train_df)
        #self.train_tokenized_x = self._tokenize(self.train_df_x)
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(self.train_df_x,
                                                                                        self.train_df_y,
                                                                                        test_size=self.config[
                                                                                            'split_ratio'],
                                                                                        random_state=self.config[
                                                                                            'random_seed'])

        #self.validate_y = np.delete(self.validate_y, -1, 1)  # delete the end 'none' at axis = 1

        test_df = pd.read_csv(self.config['input_testset'])

        self.test_x, self.test_ids = self._parse(test_df, is_test=True)
        #self.test_x = self._test_tokenize(self.test_x)

    @staticmethod
    def clean_txt(text):
        text = text.strip().lower().replace('\n', '')
        text = text.replace(r"\d", "")
        text = text.replace(r'(([0-9]{1,}\.){2,}[0-9]{1,})', ' ')
        text = text.replace(r"[^a-zA-Z0-9.,\"!]+", " ")
        text = text.replace("'", "")
        text = text.replace(r"\\n{1,}", " line ")
        #text = Preprocessor._normalize(text)
        #filter_table = str.maketrans('', '', string.punctuation)
        #clean_words = [w.translate(filter_table) for w in text if len(w.translate(filter_table))]
        return text


    def _normalize(text):
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        stop_words = set(stopwords.words('english'))
        # one to one mapping the filters characters to its len's " "
        translate_map = str.maketrans(filters, " " * len(filters))

        text = text.lower()
        text = text.translate(translate_map)

        tokens = nltk.word_tokenize(text)

        tags = nltk.pos_tag(tokens)

        normalized_text = [WordNetLemmatizer().lemmatize(tag[0], pos=Preprocessor._get_part_of_speech(
            tag[1])) for tag in tags if tag[0] not in stop_words if len(tag[0]) > 2]

        return normalized_text


    def _get_part_of_speech(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def process(self):
        input_convertor = self.config.get('input_convertor', None)
        label_convertor = self.config.get('label_convertor', None)
        train_df_x, train_df_y, train_x, train_y, validate_x, validate_y, test_x = \
            self.train_df_x, self.train_df_y, self.train_x, self.train_y, \
            self.validate_x, self.validate_y, self.test_x

        if input_convertor == 'count_vectorization':
            train_x, validate_x, test_x = self.count_vectorization(train_x, validate_x, test_x)
            # data_x, test_x = self.count_vectorization(data_x, test_x)
        elif input_convertor == 'tfidf_vectorization':
            train_x, validate_x, test_x = self.tfidf_vectorization(train_x, validate_x, test_x)
            # data_x, test_x = self.tfidf_vectorization(data_x, test_x)
        elif input_convertor == 'nn_vectorization':  # for neural network
            train_x, train_df_x, validate_x, test_x = self.nn_vectorization(train_x, train_df_x, validate_x, test_x)

            # data_x, test_x = self.nn_vectorization(data_x, test_x)
        # print(train_x.shape)
        # print(train_y.shape)
        # print(validate_x.shape)
        # print(validate_y.shape)
        return train_df_x, train_df_y, train_x, train_y, validate_x, validate_y, test_x

    # def count_vectorization(self, train, test):
    #     vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    #     vectorized_train_x = vectorizer.fit_transform(train)
    #     vectorized_test_x = vectorizer.transform(test)
    #     return vectorized_train_x, vectorized_test_x
    #
    #
    # def tfidf_vectorization(self, train, test):
    #     vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    #     vectorized_train_x = vectorizer.fit_transform(train)
    #     vectorized_test_x = vectorizer.transform(test)
    #     return vectorized_train_x, vectorized_test_x

    def count_vectorization(self, train_x, validate_x, test_x):
        vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x

    def tfidf_vectorization(self, train_x, validate_x, test_x):
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_train_x = vectorizer.fit_transform(train_x)
        vectorized_validate_x = vectorizer.transform(validate_x)
        vectorized_test_x = vectorizer.transform(test_x)
        return vectorized_train_x, vectorized_validate_x, vectorized_test_x


    def nn_vectorization(self, train_x, train_full_x, validate_x, test_x):
        self.word2ind = {}
        self.ind2word = {}

        specialtokens = ['<pad>','<unk>']

        pretrained_embedding = self.config.get('embedding_file_input', None)

        if pretrained_embedding is not None:
            word2embedding = Preprocessor_cnn.load_vector(pretrained_embedding)
            print(type(word2embedding))
            vocab = specialtokens + list(word2embedding.keys())
            vocab_size = len(vocab)
            print("vocab_size")
            print(vocab_size)
            print("word2embedding size")

            x_tokenizer = text.Tokenizer(vocab_size)
            self.embedding_matrix = np.zeros((vocab_size, self.config['embedding_dim']))
            for token in specialtokens:
                word2embedding[token] = np.random.uniform(low=-1, high=1,
                                size=(self.config['embedding_dim']))
            #print(word2embedding)
            for idx, word in enumerate(vocab):

                self.word2ind[word] = idx
                self.ind2word[idx] = word
                #print(word)
                self.embedding_matrix[idx] = word2embedding[word]

        else:
            def addword(word2ind,ind2word,word):
                if word in word2ind:
                    return
                ind2word[len(word2ind)] = word
                word2ind[word] = len(word2ind)

            for token in specialtokens:
                addword(self.word2ind, self.ind2word, token)

            for sent in train_x:
                for word in sent:
                    addword(self.word2ind, self.ind2word, word)

        train_x_ids = []
        for sent in train_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            train_x_ids.append(indsent)

        train_x_ids = np.array(train_x_ids)

        train_full_x_ids = []
        for sent in train_full_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            train_full_x_ids.append(indsent)

        train_full_x_ids = np.array(train_full_x_ids)

        validate_x_ids = []
        for sent in validate_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            validate_x_ids.append(indsent)

        validate_x_ids = np.array(validate_x_ids)

        test_x_ids = []
        for sent in test_x:
            indsent = [self.word2ind.get(i, self.word2ind['<unk>']) for i in sent]
            test_x_ids.append(indsent)

        test_x_ids = np.array(test_x_ids)

        train_x_ids = keras.preprocessing.sequence.pad_sequences(train_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
        validate_x_ids = keras.preprocessing.sequence.pad_sequences(validate_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
        test_x_ids = keras.preprocessing.sequence.pad_sequences(test_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])
        train_full_x_ids = keras.preprocessing.sequence.pad_sequences(train_full_x_ids, maxlen=self.config['maxlen'], padding='post',value=self.word2ind['<pad>'])

        return train_x_ids, train_full_x_ids, validate_x_ids, test_x_ids

    @staticmethod
    def load_vector(embedding_input):
        embeddings_index = {}
        f = io.open(embedding_input, 'r', encoding='utf-8')
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            coef = np.asarray(tokens[1:], dtype='float32')
            # coef = np.array(list(map(float, tokens[1:]))
            embeddings_index[word] = coef
        print(f'Found {len(embeddings_index)} word vectors.')
        return embeddings_index