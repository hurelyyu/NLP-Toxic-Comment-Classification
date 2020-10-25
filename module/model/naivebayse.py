import numpy as np
from sklearn.naive_bayes import MultinomialNB

class NaiveBayer(object):
    def __init__(self, classes):
        self.models = {}
        self.classes = classes
        for cls in self.classes:
            model = MultinomialNB()
            self.models[cls] = model

    def fit(self, train_x, train_y):
        # enumerate :https://www.geeksforgeeks.org/enumerate-in-python/
        for idx, cls in enumerate(self.classes):
            class_labels = train_y[:,idx]
            self.models[cls].fit(train_x, class_labels)

    def predict(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.models[cls].predict(test_x)
        return predictions

    def predict_prob(self, test_x):
        probs = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            # only want the probability of getting the output either as 0 or 1
            # Ref: https://discuss.analyticsvidhya.com/t/what-is-the-difference-between-predict-and-predict-proba/67376/3
            probs[:, idx] = self.models[cls].predict_proba(test_x)[:,1]
        return probs
