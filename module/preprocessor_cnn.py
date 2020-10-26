import re, os, sys, string, itertools
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
        x_test_tokenized=x_tokenizer.texts_to_sequences(test)
        test = sequence.pad_sequences(x_test_tokenized,maxlen=self.config['max_text_length'])

        return test

    def _load_data(self):
        train_df = pd.read_csv(self.config['input_trainset'])
        train_df['none'] = 1 - train_df[self.classes].max(axis=1)
        self.train_df_x, self.train_df_y = self._parse(train_df)
        self.train_tokenized_x = self._tokenize(self.train_df_x)
        self.train_x, self.validate_x, self.train_y, self.validate_y = train_test_split(self.train_tokenized_x,
                                                                                        self.train_df_y,
                                                                                        test_size=self.config[
                                                                                            'split_ratio'],
                                                                                        random_state=self.config[
                                                                                            'random_seed'])

        self.validate_y = np.delete(self.validate_y, -1, 1)  # delete the end 'none' at axis = 1

        test_df = pd.read_csv(self.config['input_testset'])

        self.test_x, self.test_ids = self._parse(test_df, is_test=True)
        self.test_x = self._test_tokenize(self.test_x)

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
        # input_convertor = self.config.get('input_convertor', None)
        # label_convertor = self.config.get('label_convertor', None)
        train_df_x, train_df_y, train_x, train_y, validate_x, validate_y, test_x = \
            self.train_df_x, self.train_df_y, self.train_x, self.train_y, \
            self.validate_x, self.validate_y, self.test_x

        # if input_convertor == 'count_vectorization':

        #     train_x, validate_x = self.count_vectorization(train_x, validate_x)
        #     train_df_x, test_x = self.count_vectorization(train_df_x, test_x)

        # elif input_convertor == 'tfidf_vectorization':

        #     train_x, validate_x = self.tfidf_vectorization(train_x, validate_x)
        #     train_df_x, test_x = self.tfidf_vectorization(train_df_x, test_x)

        return train_df_x, train_df_y, train_x, train_y, validate_x, validate_y, test_x


    # def count_vectorization(self, train, test):
    #     vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    #     vectorized_train_x = vectorizer.fit_transform(train)
    #     vectorized_test_x = vectorizer.transform(test)
    #     return vectorized_train_x, vectorized_test_x


    # def tfidf_vectorization(self, train, test):
    #     vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    #     vectorized_train_x = vectorizer.fit_transform(train)
    #     vectorized_test_x = vectorizer.transform(test)
    #     return vectorized_train_x, vectorized_test_x
